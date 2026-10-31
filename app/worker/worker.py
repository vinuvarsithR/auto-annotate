"""
worker.py - background worker that runs detect jobs.

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
* run_once() processes a single job without a thread (used by tests).
* The model code is loaded lazily through load_pipeline(), so the API starts
  even when autolabel.models is not installed, and tests can pass a fake.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Callable

from app.db import db

log = logging.getLogger("app.worker")

# load_pipeline() -> (run_function, ClassSpec class)
PipelineLoader = Callable[[], tuple[Callable, type]]


def default_load_pipeline() -> tuple[Callable, type]:
    from autolabel import pipeline
    from autolabel.models.base import ClassSpec
    return pipeline.run, ClassSpec


def one_line(exc: BaseException, limit: int = 300) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    text = text.splitlines()[0] if text else type(exc).__name__
    return text[:limit]


class Worker:
    def __init__(self, db_path: str | Path, load_pipeline: PipelineLoader | None = None,
                 poll_seconds: float = 2.0):
        self.db_path = db_path
        self.load_pipeline = load_pipeline or default_load_pipeline
        self.poll_seconds = poll_seconds
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------ thread control

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="detect-worker", daemon=True)
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
        """Run the oldest queued detect job. Returns False if there was none."""
        conn = db.connect(self.db_path)
        try:
            job = self._claim(conn)
            if job is None:
                return False
            try:
                self._process(conn, job)
            except Exception as exc:
                log.warning("job %s failed: %s", job["id"], one_line(exc))
                conn.rollback()
                with conn:
                    conn.execute("UPDATE jobs SET status='failed', error=?, finished_at=? "
                                 "WHERE id=?", (one_line(exc), db.now(), job["id"]))
            return True
        finally:
            conn.close()

    def _claim(self, conn: sqlite3.Connection) -> sqlite3.Row | None:
        conn.execute("BEGIN IMMEDIATE")  # lock so two workers can't take the same job
        try:
            job = conn.execute("SELECT * FROM jobs WHERE status='queued' AND kind='detect' "
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
