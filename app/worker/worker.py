"""
worker.py - background worker that runs detect and roboflow_push jobs.

Plain-English summary
---------------------
* One thread. It takes the oldest 'queued' detect job, marks it 'running',
  and calls autolabel.pipeline.run(...) on the job's images.
* pipeline.run yields one image at a time. For each image, in ONE transaction:
    1. delete that image's earlier MODEL boxes that are still auto_accepted / needs_review
       (human-reviewed rows are never touched),
    1b. drop new boxes that overlap a remaining non-rejected box of the same class at
       IoU >= 0.5 (#9, pipeline.drop_overlapping), so imported / human / reviewed boxes
       are not duplicated,
    2. store the new boxes: score >= class accept_score -> auto_accepted, else needs_review
       (boxes below the class min_score are dropped, as a safety net),
    3. update done / total / progress.
  So if the job fails halfway, images that weren't reached keep their old boxes.
* Examples mode (#9, config.mode 'examples'): first the project's reviewed boxes are read,
  cleaned and capped (read_examples) and pipeline.prepare_examples embeds them, then `result`
  is written, then the images run as above with only the classes that have usable examples.
  Earlier model boxes of the other classes are kept. No class left -> done, nothing run.
* Any exception -> job 'failed' with a one-line error message.
* roboflow_push jobs (#21): the images with review_status done/empty and no
  roboflow_image_id yet, one at a time: push(image, boxes, class_names, batch_name),
  where each box has class_name, bbox and polygon, returns the Roboflow id, which is saved at once (so a failed job can just be
  started again). Errors become a one-line message with the API key removed.
* import jobs (#22): the upload in data/imports/<job_id>/ (a zip is extracted first) is
  read by autolabel.importers.plan_import (export's code). Missing classes are added
  at the end of the class list, then image by image, in ONE transaction each: the #4
  duplicate rule (db.insert_image), the image's boxes (with their polygons) as
  needs_review / source 'import', done / progress. At the end `result` is stored and the folder is deleted (also when
  the job fails).
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
import shutil
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

# load_importers() -> the autolabel.importers module (or a stub with the same names)
ImporterLoader = Callable[[], object]

EXAMPLE_STATUSES = ("accepted", "edited")  # reviewed boxes teach the next examples-mode run
EXAMPLE_CAP = 30  # per class; fixed by the contract, not an option
PUSH_REVIEW_STATUSES = ("done", "empty")
DEFAULT_PUSH_STATUSES = ("auto_accepted", "accepted", "edited")


def default_load_pipeline() -> tuple[Callable, type]:
    from autolabel import pipeline
    from autolabel.models.base import ClassSpec
    return pipeline.run, ClassSpec


def default_load_importers():
    import autolabel.importers as importers
    return importers


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


def clip_box(x: float, y: float, w: float, h: float, width: int, height: int) -> tuple | None:
    """(x, y, w, h) clipped to the image, or None if no area is left."""
    x1, y1 = max(x, 0.0), max(y, 0.0)
    x2, y2 = min(x + w, float(width)), min(y + h, float(height))
    if x2 > x1 and y2 > y1:
        return x1, y1, x2 - x1, y2 - y1
    return None


def evenly_spaced(rows: list, cap: int) -> list:
    """At most `cap` items, picked at even steps through `rows`, order kept."""
    n = len(rows)
    return rows if n <= cap else [rows[i * n // cap] for i in range(cap)]


def read_examples(conn: sqlite3.Connection, pid: int, example_type: type) -> dict[str, list]:
    """Class name -> its examples, from all the project's images (#9, examples mode).

    Only the bbox of a row is used. A box with no area left after clipping, or whose image
    file is missing, is not an example. The cap comes last, so those do not use up places.

    `labelled` of each example: every other non-rejected box of the same class on the same
    image (any source or status, cap ignored, same clipping), so owlv2 does not count them as
    background. The example's own box is in `bbox`, so it is left out of `labelled`.
    """
    rows = conn.execute(
        "SELECT a.id, a.x, a.y, a.w, a.h, a.status, c.name AS class_name, i.id AS image_id, "
        "i.path, i.width, i.height FROM annotations a JOIN classes c ON c.id = a.class_id "
        "JOIN images i ON i.id = a.image_id WHERE i.project_id=? AND a.status != 'rejected' "
        "ORDER BY a.id", (pid,))
    has_file: dict[int, bool] = {}
    by_class: dict[str, list] = {}  # class -> [(annotation id, row, clipped box)] of examples
    labelled: dict[tuple, list] = {}  # (class, image id) -> [(annotation id, clipped box)]
    for r in rows:
        box = clip_box(r["x"], r["y"], r["w"], r["h"], r["width"], r["height"])
        if box is None:
            continue
        labelled.setdefault((r["class_name"], r["image_id"]), []).append((r["id"], box))
        if r["status"] not in EXAMPLE_STATUSES:
            continue
        if r["image_id"] not in has_file:
            has_file[r["image_id"]] = Path(r["path"]).is_file()
        if has_file[r["image_id"]]:
            by_class.setdefault(r["class_name"], []).append((r["id"], r, box))
    return {name: [example_type(image_id=str(r["image_id"]), path=r["path"],
                                category=name, bbox=box,
                                labelled=tuple(b for aid, b in labelled[(name, r["image_id"])]
                                               if aid != own_id))
                   for own_id, r, box in evenly_spaced(found, EXAMPLE_CAP)]
            for name, found in by_class.items()}


class Worker:
    def __init__(self, db_path: str | Path, load_pipeline: PipelineLoader | None = None,
                 poll_seconds: float = 2.0, load_pusher: PusherLoader | None = None,
                 load_importers: ImporterLoader | None = None):
        self.db_path = db_path
        self.load_pipeline = load_pipeline or default_load_pipeline
        self.load_pusher = load_pusher or default_load_pusher
        self.load_importers = load_importers or default_load_importers
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
                elif job["kind"] == "import":
                    self._process_import(conn, job)
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

        # Text jobs and old jobs (no config.mode) call run with exactly these arguments.
        run_args = {"detector": model, "options": options}
        delete_filter, delete_args = "", []
        if config.get("mode") == "examples":
            example_run = self._prepare_examples(conn, jid, pid, model, options, specs)
            if example_run is None:  # no class left: nothing is run or deleted
                with conn:
                    conn.execute("UPDATE jobs SET status='done', progress=1.0, done=?, "
                                 "finished_at=? WHERE id=?", (total, db.now(), jid))
                return
            run_args["example_run"] = example_run
            specs = list(example_run.classes)
            by_name = {s.name: by_name[s.name] for s in specs}  # a skipped class must not come back
            delete_args = [r["id"] for r in by_name.values()]
            delete_filter = f" AND class_id IN ({','.join('?' * len(delete_args))})"

        from autolabel.evaluate import Box  # pure functions, no torch
        from autolabel.pipeline import drop_overlapping

        expected = {img_id for img_id, _ in images}
        done = 0
        for image_id, preds in run(images, specs, **run_args):
            if image_id not in expected:
                raise ValueError(f"detector returned unknown image_id '{image_id}'")
            preds = list(preds)
            for p in preds:  # unknown class -> job fails, even if the box would be dropped
                self._row(p, by_name, image_id)
            done += 1
            t = db.now()
            with conn:
                conn.execute(
                    "DELETE FROM annotations WHERE image_id=? "
                    "AND source NOT IN ('human', 'import') "
                    "AND status IN (?, ?)" + delete_filter,
                    (int(image_id), *db.MODEL_PENDING, *delete_args))
                # #9: drop new boxes that overlap a remaining non-rejected box of the same class
                existing = [
                    Box(image_id, r["class_name"],
                        (float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])))
                    for r in conn.execute(db.ANNOTATION_SELECT + " WHERE a.image_id=? "
                                          "AND a.status != 'rejected'", (int(image_id),))]
                kept = drop_overlapping(preds, existing) if existing else preds
                rows_to_add = [self._row(p, by_name, image_id) for p in kept]
                for r in rows_to_add:
                    if r is None:
                        continue
                    conn.execute(
                        "INSERT INTO annotations (image_id, class_id, x, y, w, h, mask, polygon, "
                        "score, source, status, job_id, reviewer, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)",
                        (int(image_id), r["class_id"], *r["bbox"], r["mask"], r["polygon"],
                         r["score"], r["source"] or model, r["status"], jid, t, t))
                conn.execute("UPDATE jobs SET done=?, progress=? WHERE id=?",
                             (done, done / total if total else 1.0, jid))

        with conn:
            conn.execute("UPDATE jobs SET status='done', progress=1.0, finished_at=? WHERE id=?",
                         (db.now(), jid))

    def _prepare_examples(self, conn: sqlite3.Connection, jid: int, pid: int, model: str,
                          options: dict, specs: list):
        """Steps 1-3 of #9 examples mode. Returns the ExampleRun, or None if no class is left."""
        from autolabel import pipeline
        from autolabel.models.base import Example

        examples = read_examples(conn, pid, Example)
        ready = [s for s in specs if s.name in examples]
        skipped = [{"class_name": s.name, "reason": "no examples"}
                   for s in specs if s.name not in examples]
        example_run, used = None, {}
        if ready:  # slow: the job stays 'running' at progress 0 while it embeds
            example_run = pipeline.prepare_examples(
                model, ready, {s.name: examples[s.name] for s in ready}, options)
            used = dict(example_run.used)
            # the detector may say why (e.g. owlv2 "needs at least 5 examples"); fake/yoloe don't
            reasons = getattr(example_run, "skip_reasons", None) or {}
            skipped += [{"class_name": name, "reason": reasons.get(name) or "no usable examples"}
                        for name in example_run.unusable]
        result = {"mode": "examples", "examples_by_class": used, "skipped_classes": skipped}
        with conn:
            conn.execute("UPDATE jobs SET result=? WHERE id=?", (json.dumps(result), jid))
        return example_run if example_run is not None and example_run.classes else None

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
                    {k: a[k] for k in ("class_name", "bbox", "polygon")}
                    for a in map(db.annotation_dict, conn.execute(
                        db.ANNOTATION_SELECT + f" WHERE a.image_id=? AND a.status IN ({marks}) "
                        "ORDER BY a.id", (img["id"], *statuses)))]
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

    def imports_dir(self) -> Path:
        return Path(self.db_path).parent / "imports"

    def _process_import(self, conn: sqlite3.Connection, job: sqlite3.Row) -> None:
        """#22: add the images and boxes of an uploaded zip / folder. Folder deleted at the end."""
        folder = self.imports_dir() / str(job["id"])
        try:
            self._run_import(conn, job, folder)
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    def _run_import(self, conn: sqlite3.Connection, job: sqlite3.Row, folder: Path) -> None:
        from app.api.images import read_image_size  # same JPEG/PNG check as #4
        from app.api.imports import max_import_bytes

        jid, pid = job["id"], job["project_id"]
        config = json.loads(job["config"])
        if not folder.is_dir():
            raise FileNotFoundError(f"upload folder of import job {jid} is missing")
        imp = self.load_importers()
        root = folder / "files"
        skipped: list[dict] = []
        if config.get("upload") == "zip":
            skipped += list(imp.extract_zip(folder / "upload.zip", root, max_import_bytes()))
        root.mkdir(parents=True, exist_ok=True)

        plan = imp.plan_import(root)
        skipped += list(plan.skipped)
        items = list(plan.items)
        total = len(items)
        with conn:
            conn.execute("UPDATE jobs SET total=? WHERE id=?", (total, jid))

        # classes: exact-name match; missing names go at the end, in file order
        have = {r["name"] for r in conn.execute(
            "SELECT name FROM classes WHERE project_id=?", (pid,))}
        created = [n for n in dict.fromkeys(plan.class_names) if n not in have]
        with conn:
            for name in created:
                conn.execute("INSERT INTO classes (project_id, name, description, min_score, "
                             "accept_score) VALUES (?, ?, ?, NULL, NULL)", (pid, name, name))
        class_ids = {r["name"]: r["id"] for r in conn.execute(
            "SELECT id, name FROM classes WHERE project_id=?", (pid,))}

        uploads = self.imports_dir().parent / "uploads" / str(pid)
        uploads.mkdir(parents=True, exist_ok=True)
        added = 0
        duplicates: list[str] = []
        renamed: list[dict] = []
        boxes_by_class: dict[str, int] = {}

        for done, item in enumerate(items, start=1):
            name = item.rel_path.replace("\\", "/").rsplit("/", 1)[-1]
            try:
                data = (root / item.rel_path).read_bytes()
            except OSError:
                data = None
            size = read_image_size(data, name) if data is not None else None
            progress = done / total
            if size is None:
                skipped.append({"file_name": item.rel_path, "reason": "not an image"})
                with conn:
                    conn.execute("UPDATE jobs SET done=?, progress=? WHERE id=?",
                                 (done, progress, jid))
                continue

            conn.execute("BEGIN IMMEDIATE")  # same lock as #4 uploads
            target = None
            stored: list[str] = []
            try:
                row = db.insert_image(conn, pid, uploads, name, data, size)
                if row is None:
                    duplicates.append(item.rel_path)  # its boxes are not imported
                else:
                    iid, final, target = row
                    if final != name:
                        renamed.append({"from": name, "to": final})
                    t = db.now()
                    for box in item.boxes:
                        x, y, w, h = (float(v) for v in box["bbox"])
                        if w <= 0 or h <= 0:
                            continue
                        polygon = box.get("polygon")  # plan_import already set bbox from it
                        conn.execute(
                            "INSERT INTO annotations (image_id, class_id, x, y, w, h, mask, "
                            "polygon, score, source, status, job_id, reviewer, created_at, "
                            "updated_at) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, 'import', "
                            "'needs_review', ?, NULL, ?, ?)",
                            (iid, class_ids[box["class_name"]], x, y, w, h,
                             json.dumps(polygon) if polygon is not None else None, jid, t, t))
                        stored.append(box["class_name"])
                conn.execute("UPDATE jobs SET done=?, progress=? WHERE id=?",
                             (done, progress, jid))
                conn.commit()
            except BaseException:
                conn.rollback()
                if target is not None:
                    target.unlink(missing_ok=True)
                raise
            if row is not None:
                added += 1
                for n in stored:
                    boxes_by_class[n] = boxes_by_class.get(n, 0) + 1

        result = {"format": plan.format, "images_added": added, "duplicates": duplicates,
                  "renamed": renamed, "boxes_by_class": boxes_by_class,
                  "classes_created": created, "skipped": skipped}
        with conn:
            conn.execute("UPDATE jobs SET status='done', progress=1.0, done=?, result=?, "
                         "finished_at=? WHERE id=?", (total, json.dumps(result), db.now(), jid))

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
        polygon = getattr(pred, "polygon", None)
        return {
            "class_id": cls["id"],
            "bbox": tuple(float(v) for v in pred.bbox),
            "mask": json.dumps(mask) if mask is not None else None,
            "polygon": json.dumps(polygon) if polygon is not None else None,
            "score": score,
            "source": getattr(pred, "source", "") or "",
            "status": status,
        }
