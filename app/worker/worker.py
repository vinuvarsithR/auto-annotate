"""
worker.py - background worker that runs detect and roboflow_push jobs.

Plain-English summary
---------------------
* One thread. It takes the oldest 'queued' detect job, marks it 'running',
  and calls autolabel.pipeline.run(...) on the job's images.
* pipeline.run yields one image at a time. For each image, in ONE transaction:
    1. delete that image's earlier MODEL boxes that are still auto_accepted / needs_review
       (human-reviewed rows are never touched),
    2. store the new boxes: score >= class accept_score -> auto_accepted, else needs_review
       (boxes below the class min_score are dropped, as a safety net),
    3. update done / total / progress.
  So if the job fails halfway, images that weren't reached keep their old boxes.
* Any exception -> job 'failed' with a one-line error message.
* roboflow_push jobs (#21): the images with review_status done/empty and no
  roboflow_image_id yet, one at a time: push(image, boxes, class_names, batch_name)
  returns the Roboflow id, which is saved at once (so a failed job can just be
  started again). Errors become a one-line message with the API key removed.
* run_once() processes a single job without a thread (used by tests).
* The model code is loaded lazily through load_pipeline(), and the Roboflow
  code through load_pusher(), so the API starts without them and tests can
  pass fakes (no network in tests).
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from app.db import db

log = logging.getLogger("app.worker")

# load_pipeline() -> (run_function, ClassSpec class)
PipelineLoader = Callable[[], tuple[Callable, type]]

# load_pusher() -> (push, secrets)
#   push(image: dict, boxes: list[dict], class_names: list[str], batch_name: str) -> str
#   secrets: strings that must never appear in a job error (the API key)
PusherLoader = Callable[[], tuple[Callable[..., str], list[str]]]

PUSH_REVIEW_STATUSES = ("done", "empty")
DEFAULT_PUSH_STATUSES = ("auto_accepted", "accepted", "edited")


def default_load_pipeline() -> tuple[Callable, type]:
    from autolabel import pipeline
    from autolabel.models.base import ClassSpec
    return pipeline.run, ClassSpec


def default_load_pusher() -> tuple[Callable[..., str], list[str]]:
    """Bind autolabel.export.push_image to the .env settings (export session's code)."""
    from autolabel.export import push_image, roboflow_settings
    settings = roboflow_settings()
    if isinstance(settings, (list, tuple)):  # contract: the list of missing names
        raise RuntimeError("Roboflow settings missing from .env: " + ", ".join(settings))
    secrets = [os.environ.get("ROBOFLOW_API_KEY", "")]
    if isinstance(settings, dict):
        secrets += [str(v) for k, v in settings.items() if "key" in str(k).lower()]

    def push(image, boxes, class_names, batch_name):
        return push_image(settings, image, boxes, class_names, batch_name)
    return push, [x for x in secrets if x]


_KEY_PARAM = re.compile(r"(api_key=)[^&\s'\"]+", re.IGNORECASE)


def one_line(exc: BaseException, limit: int = 300, secrets=()) -> str:
    """First line of the error, key removed, at most `limit` chars."""
    text = f"{type(exc).__name__}: {exc}".strip()
    text = text.splitlines()[0] if text else type(exc).__name__
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    text = _KEY_PARAM.sub(r"\1***", text)
    return text[:limit]


class Worker:
    def __init__(self, db_path: str | Path, load_pipeline: PipelineLoader | None = None,
                 poll_seconds: float = 2.0, load_pusher: PusherLoader | None = None):
        self.db_path = db_path
        self.load_pipeline = load_pipeline or default_load_pipeline
        self.load_pusher = load_pusher or default_load_pusher
        self._secrets: list[str] = []
        self.poll_seconds = poll_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ thread control

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="job-worker", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception:  # never let the thread die
                log.exception("worker loop error")
                worked = False
            if not worked:
                self._wake.wait(self.poll_seconds)
                self._wake.clear()

    # ------------------------------------------------------------ one job

    def run_once(self) -> bool:
        """Run the oldest queued job (any kind). Returns False if there was none."""
        conn = db.connect(self.db_path)
        try:
            job = self._claim(conn)
            if job is None:
                return False
            self._secrets = []
            try:
                if job["kind"] == "roboflow_push":
                    self._process_push(conn, job)
                else:
                    self._process(conn, job)
            except Exception as exc:
                msg = one_line(exc, secrets=self._secrets)
                log.warning("job %s failed: %s", job["id"], msg)
                conn.rollback()
                with conn:
                    conn.execute("UPDATE jobs SET status='failed', error=?, finished_at=? "
                                 "WHERE id=?", (msg, db.now(), job["id"]))
            return True
        finally:
            conn.close()

    def _claim(self, conn: sqlite3.Connection) -> sqlite3.Row | None:
        conn.execute("BEGIN IMMEDIATE")  # lock so two workers can't take the same job
        try:
            job = conn.execute("SELECT * FROM jobs WHERE status='queued' "
                               "ORDER BY id LIMIT 1").fetchone()
            if job is not None:
                conn.execute("UPDATE jobs SET status='running', started_at=? WHERE id=?",
                             (db.now(), job["id"]))
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        return job

    def _process(self, conn: sqlite3.Connection, job: sqlite3.Row) -> None:
        jid, pid = job["id"], job["project_id"]
        config = json.loads(job["config"])
        model = config.get("model", "pipeline")
        options = config.get("options") or {}

        if config.get("image_ids") is None:
            rows = conn.execute("SELECT id, path FROM images WHERE project_id=? ORDER BY id",
                                (pid,)).fetchall()
        else:
            wanted = set(config["image_ids"])
            rows = [r for r in conn.execute(
                "SELECT id, path FROM images WHERE project_id=? ORDER BY id", (pid,))
                if r["id"] in wanted]
        images = [(str(r["id"]), r["path"]) for r in rows]
        total = len(images)
        with conn:
            conn.execute("UPDATE jobs SET total=? WHERE id=?", (total, jid))

        class_rows = conn.execute("SELECT * FROM classes WHERE project_id=? ORDER BY id",
                                  (pid,)).fetchall()
        if not class_rows:
            raise ValueError(f"project {pid} has no classes")
        by_name = {r["name"]: r for r in class_rows}

        run, ClassSpec = self.load_pipeline()
        specs = [ClassSpec(name=r["name"], description=r["description"], min_score=r["min_score"])
                 for r in class_rows]

        expected = {img_id for img_id, _ in images}
        done = 0
        for image_id, preds in run(images, specs, detector=model, options=options):
            if image_id not in expected:
                raise ValueError(f"detector returned unknown image_id '{image_id}'")
            rows_to_add = [self._row(p, by_name, image_id) for p in preds]
            done += 1
            t = db.now()
            with conn:
                conn.execute(
                    "DELETE FROM annotations WHERE image_id=? AND source != 'human' "
                    "AND status IN (?, ?)", (int(image_id), *db.MODEL_PENDING))
                for r in rows_to_add:
                    if r is None:
                        continue
                    conn.execute(
                        "INSERT INTO annotations (image_id, class_id, x, y, w, h, mask, score, "
                        "source, status, job_id, reviewer, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                        (int(image_id), r["class_id"], *r["bbox"], r["mask"], r["score"],
                         r["source"] or model, r["status"], jid, t, t))
                conn.execute("UPDATE jobs SET done=?, progress=? WHERE id=?",
                             (done, done / total if total else 1.0, jid))

        with conn:
            conn.execute("UPDATE jobs SET status='done', progress=1.0, finished_at=? WHERE id=?",
                         (db.now(), jid))

    def _process_push(self, conn: sqlite3.Connection, job: sqlite3.Row) -> None:
        """#21: upload done/empty images not pushed yet, one at a time."""
        jid, pid = job["id"], job["project_id"]
        config = json.loads(job["config"])
        statuses = list(config.get("statuses") or DEFAULT_PUSH_STATUSES)
        if "rejected" in statuses:
            raise ValueError("rejected boxes are never pushed")
        started = conn.execute("SELECT started_at FROM jobs WHERE id=?", (jid,)).fetchone()[0]
        batch_name = f"autolabel_{(started or db.now())[:10]}"

        rows = conn.execute(
            "SELECT * FROM images WHERE project_id=? AND roboflow_image_id IS NULL "
            "AND review_status IN (?, ?) ORDER BY id", (pid, *PUSH_REVIEW_STATUSES)).fetchall()
        total = len(rows)
        with conn:
            conn.execute("UPDATE jobs SET total=? WHERE id=?", (total, jid))

        if total:
            # class order = the order #7 returns (by id)
            class_names = [r["name"] for r in conn.execute(
                "SELECT name FROM classes WHERE project_id=? ORDER BY id", (pid,))]
            push, self._secrets = self.load_pusher()
            marks = ",".join("?" * len(statuses))
            for done, img in enumerate(rows, start=1):
                boxes = [] if img["review_status"] == "empty" else [
                    {"class_name": b["class_name"],
                     "bbox": [float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])]}
                    for b in conn.execute(
                        db.ANNOTATION_SELECT + f" WHERE a.image_id=? AND a.status IN ({marks}) "
                        "ORDER BY a.id", (img["id"], *statuses))]
                image = {"id": img["id"], "file_name": img["file_name"], "path": img["path"],
                         "width": img["width"], "height": img["height"]}
                rf_id = push(image, boxes, class_names, batch_name)
                if not rf_id:
                    raise RuntimeError(f"Roboflow returned no id for image {img['id']}")
                with conn:
                    conn.execute("UPDATE images SET roboflow_image_id=? WHERE id=?",
                                 (str(rf_id), img["id"]))
                    conn.execute("UPDATE jobs SET done=?, progress=? WHERE id=?",
                                 (done, done / total, jid))

        with conn:
            conn.execute("UPDATE jobs SET status='done', progress=1.0, finished_at=? WHERE id=?",
                         (db.now(), jid))

    @staticmethod
    def _row(pred, by_name: dict, image_id: str) -> dict | None:
        """Turn one Prediction into insert values, or None if it is below min_score."""
        cls = by_name.get(pred.category)
        if cls is None:
            raise ValueError(f"detector returned unknown class '{pred.category}'")
        score = float(pred.score)
        if cls["min_score"] is not None and score < cls["min_score"]:
            return None
        accept = cls["accept_score"]
        status = "auto_accepted" if accept is not None and score >= accept else "needs_review"
        mask = getattr(pred, "mask", None)
        return {
            "class_id": cls["id"],
            "bbox": tuple(float(v) for v in pred.bbox),
            "mask": json.dumps(mask) if mask is not None else None,
            "score": score,
            "source": getattr(pred, "source", "") or "",
            "status": status,
        }
