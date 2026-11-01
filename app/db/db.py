"""
db.py - SQLite helpers for the backend.

Plain-English summary
---------------------
* One file, data/app.db. Tables are created from schema.sql the first time.
* connect() gives a connection with foreign keys ON (SQLite has them off by default).
* The *_dict() functions turn database rows into the exact JSON shapes in
  docs/CONTRACT.md section 2 (Project, Image, Class, Job, Annotation).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

SCHEMA = Path(__file__).with_name("schema.sql")

STATUSES = ("auto_accepted", "needs_review", "accepted", "rejected", "edited")
REVIEW_STATUSES = ("todo", "done", "empty")  # images.review_status, set only by #20
BAD_REVIEW_STATUS = "review_status must be one of " + ", ".join(REVIEW_STATUSES)
MODEL_PENDING = ("auto_accepted", "needs_review")  # model boxes a new job may replace

_CHUNK = 500  # stay well under SQLite's limit on '?' parameters


def now() -> str:
    """Current time as ISO-8601 UTC text, e.g. 2026-10-30T14:05:00Z."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    """Create the folder and the tables if they don't exist yet."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")  # lets the worker write while the API reads
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='projects'"
        ).fetchone()
        if not exists:
            conn.executescript(SCHEMA.read_text(encoding="utf-8"))
        migrate(conn)
        conn.commit()
    finally:
        conn.close()


def migrate(conn: sqlite3.Connection) -> None:
    """Bring a database made by an older version up to schema.sql. Never deletes data.

    | Added | Column                | Old rows get |
    | #20   | images.review_status     | 'todo'       |
    | #21   | images.roboflow_image_id | NULL         |
    | #4    | images.sha256            | hash of the file at `path` (NULL if the file is missing) |
    """
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(images)")}
    if "review_status" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN review_status TEXT NOT NULL DEFAULT 'todo' "
                     "CHECK (review_status IN ('todo', 'done', 'empty'))")
    if "roboflow_image_id" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN roboflow_image_id TEXT")
    if "sha256" not in cols:
        conn.execute("ALTER TABLE images ADD COLUMN sha256 TEXT")
    backfill_sha256(conn)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def backfill_sha256(conn: sqlite3.Connection) -> int:
    """Fill images.sha256 where it is NULL by hashing the file on disk. Returns rows filled.

    Only that column is touched; a missing or unreadable file leaves it NULL (tried again
    next startup). Old duplicates already in the database are kept as they are.
    """
    rows = conn.execute("SELECT id, path FROM images WHERE sha256 IS NULL").fetchall()
    filled = 0
    for r in rows:
        try:
            digest = sha256_hex(Path(r["path"]).read_bytes())
        except OSError:
            continue
        conn.execute("UPDATE images SET sha256=? WHERE id=?", (digest, r["id"]))
        filled += 1
    return filled


JOB_KINDS = ("detect", "roboflow_push")


def create_job(conn: sqlite3.Connection, pid: int, kind: str, config: dict, total: int,
               wake: Callable[[], None] | None = None) -> dict:
    """Insert a 'queued' job and return it as Job JSON (B3, used by #9 and #21).

    * total 0 -> the job is stored 'done' at once (progress 1, finished_at set).
    * wake: pass request.app.state.worker.wake so the worker starts right away.
    """
    if kind not in JOB_KINDS:
        raise ValueError(f"unknown job kind '{kind}'")
    t = now()
    with conn:
        if total == 0:
            cur = conn.execute(
                "INSERT INTO jobs (project_id, kind, status, progress, done, total, config, "
                "created_at, started_at, finished_at) VALUES (?, ?, 'done', 1.0, 0, 0, ?, ?, ?, ?)",
                (pid, kind, json.dumps(config), t, t, t))
        else:
            cur = conn.execute(
                "INSERT INTO jobs (project_id, kind, status, progress, done, total, config, "
                "created_at) VALUES (?, ?, 'queued', 0, 0, ?, ?, ?)",
                (pid, kind, total, json.dumps(config), t))
    if wake is not None and total:
        wake()
    return job_dict(conn.execute("SELECT * FROM jobs WHERE id=?", (cur.lastrowid,)).fetchone())


def fail_stale_jobs(conn: sqlite3.Connection) -> int:
    """Jobs still 'running' at startup died with the old server: mark them failed."""
    with conn:
        cur = conn.execute(
            "UPDATE jobs SET status='failed', error='server restarted while the job was running', "
            "finished_at=? WHERE status='running'",
            (now(),),
        )
    return cur.rowcount


# ---------------------------------------------------------------- counts

def zero_counts() -> dict[str, int]:
    return {s: 0 for s in STATUSES}


def image_counts(conn: sqlite3.Connection, image_ids: list[int]) -> dict[int, dict[str, int]]:
    """Box counts per status for each image id."""
    out = {i: zero_counts() for i in image_ids}
    for start in range(0, len(image_ids), _CHUNK):
        chunk = image_ids[start:start + _CHUNK]
        marks = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT image_id, status, COUNT(*) AS n FROM annotations "
            f"WHERE image_id IN ({marks}) GROUP BY image_id, status",
            chunk,
        )
        for r in rows:
            out[r["image_id"]][r["status"]] = r["n"]
    return out


def project_counts(conn: sqlite3.Connection, pid: int) -> dict[str, int]:
    counts = {"images": conn.execute(
        "SELECT COUNT(*) FROM images WHERE project_id=?", (pid,)).fetchone()[0]}
    counts.update(zero_counts())
    rows = conn.execute(
        "SELECT a.status, COUNT(*) AS n FROM annotations a JOIN images i ON i.id = a.image_id "
        "WHERE i.project_id=? GROUP BY a.status",
        (pid,),
    )
    for r in rows:
        counts[r["status"]] = r["n"]
    return counts


# ---------------------------------------------------------------- row -> JSON

def project_dict(conn: sqlite3.Connection, row: sqlite3.Row) -> dict:
    return {"id": row["id"], "name": row["name"], "created_at": row["created_at"],
            "counts": project_counts(conn, row["id"])}


def image_dict(row: sqlite3.Row, counts: dict[str, int] | None = None) -> dict:
    return {
        "id": row["id"], "project_id": row["project_id"], "file_name": row["file_name"],
        "width": row["width"], "height": row["height"],
        "url": f"/api/images/{row['id']}/file", "uploaded_at": row["uploaded_at"],
        "review_status": row["review_status"],
        "counts": counts if counts is not None else zero_counts(),
    }


def images_dicts(conn: sqlite3.Connection, rows: list[sqlite3.Row]) -> list[dict]:
    counts = image_counts(conn, [r["id"] for r in rows])
    return [image_dict(r, counts[r["id"]]) for r in rows]


def class_dict(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "name": row["name"], "description": row["description"],
            "min_score": row["min_score"], "accept_score": row["accept_score"]}


def job_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"], "project_id": row["project_id"], "kind": row["kind"],
        "status": row["status"], "progress": round(float(row["progress"]), 4),
        "done": row["done"], "total": row["total"], "error": row["error"],
        "config": json.loads(row["config"]),
        "created_at": row["created_at"], "started_at": row["started_at"],
        "finished_at": row["finished_at"],
    }


ANNOTATION_SELECT = ("SELECT a.*, c.name AS class_name FROM annotations a "
                     "JOIN classes c ON c.id = a.class_id")


def annotation_dict(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"], "image_id": row["image_id"], "class_id": row["class_id"],
        "class_name": row["class_name"],
        "bbox": [float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])],
        "mask": json.loads(row["mask"]) if row["mask"] else None,
        "score": row["score"], "source": row["source"], "status": row["status"],
        "job_id": row["job_id"], "reviewer": row["reviewer"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def get_annotation(conn: sqlite3.Connection, aid: int) -> sqlite3.Row | None:
    return conn.execute(ANNOTATION_SELECT + " WHERE a.id=?", (aid,)).fetchone()
