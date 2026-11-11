"""
jobs.py - endpoints 9-10: queue a detect job and read its status.

The request only stores a 'queued' row. The background worker (app/worker/)
picks it up, runs the detector and fills in progress / status.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError, not_found
from app.db import db

router = APIRouter()


class JobIn(BaseModel):
    model: str = "pipeline"
    mode: Any = None  # not str: a number or list must be unknown_mode, not a generic 400
    image_ids: list[int] | None = None
    options: dict = Field(default_factory=dict)
    rerun: bool = False  # #9 "Re-run with new examples"; never stored in config


RERUN_OTHER_FIELDS = ("model", "mode", "image_ids", "options")


def _rerun_source(conn: sqlite3.Connection, pid: int) -> tuple[int, dict]:
    """#9 rerun: the project's newest detect job with config.mode 'examples' (any status)."""
    for row in conn.execute("SELECT id, config FROM jobs WHERE project_id=? AND kind='detect' "
                            "ORDER BY id DESC", (pid,)):
        try:
            config = json.loads(row["config"] or "{}")
        except ValueError:
            continue
        if isinstance(config, dict) and config.get("mode") == "examples":
            return row["id"], config
    raise ApiError(409, "conflict", f"project {pid} has no examples-mode job to re-run")


def _check_model(request: Request, model: str) -> None:
    try:
        available = list(request.app.state.list_detectors())
    except ImportError:
        available = []
    if model not in available:
        msg = (f"unknown model '{model}', available: {', '.join(available)}" if available
               else f"model '{model}' is not available (autolabel.models is not installed)")
        raise ApiError(400, "unknown_model", msg)


def _check_classes(conn: sqlite3.Connection, pid: int) -> None:
    if not conn.execute("SELECT 1 FROM classes WHERE project_id=?", (pid,)).fetchone():
        raise ApiError(400, "no_classes", f"project {pid} has no classes; PUT the classes first")


def _insert_job(request: Request, conn: sqlite3.Connection, pid: int, total: int,
                config: dict) -> dict:
    with conn:
        cur = conn.execute(
            "INSERT INTO jobs (project_id, kind, status, progress, done, total, config, created_at) "
            "VALUES (?, 'detect', 'queued', 0, 0, ?, ?, ?)",
            (pid, total, json.dumps(config), db.now()),
        )
    worker = getattr(request.app.state, "worker", None)
    if worker is not None:
        worker.wake()
    return db.job_dict(conn.execute("SELECT * FROM jobs WHERE id=?", (cur.lastrowid,)).fetchone())


def _create_rerun(pid: int, body: JobIn, request: Request, conn: sqlite3.Connection) -> dict:
    """#9 rerun: same detector, mode and options as the last examples job, on the todo images."""
    extra = [f for f in RERUN_OTHER_FIELDS if f in body.model_fields_set]
    if extra:
        raise ApiError(400, "bad_request", f"rerun takes no other fields; got: {', '.join(extra)}")
    source_id, source = _rerun_source(conn, pid)
    model = source.get("model", "pipeline")
    _check_model(request, model)
    mode = _resolve_mode(request, model, "examples")
    _check_classes(conn, pid)
    image_ids = [r[0] for r in conn.execute(
        "SELECT id FROM images WHERE project_id=? AND review_status='todo' ORDER BY id", (pid,))]
    if not image_ids:
        raise ApiError(409, "conflict", f"project {pid} has no images with review_status todo")
    config = {"model": model, "mode": mode, "image_ids": image_ids,
              "options": source.get("options") or {}, "rerun_of": source_id}
    return _insert_job(request, conn, pid, len(image_ids), config)


def _resolve_mode(request: Request, model: str, mode: Any) -> str | None:
    """#9: no mode -> the detector's first #16 mode (or None); else it must be in the list."""
    modes = list(request.app.state.detector_modes(model))
    if mode is None:
        return modes[0] if modes else None
    if mode not in modes:
        raise ApiError(400, "unknown_mode",
                       f"unknown mode {mode!r} for '{model}', available: {', '.join(modes)}")
    return mode


@router.post("/projects/{pid}/jobs", status_code=202)
def create_job(pid: int, body: JobIn, request: Request,
               conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    if body.rerun:
        return _create_rerun(pid, body, request, conn)

    _check_model(request, body.model)
    mode = _resolve_mode(request, body.model, body.mode)
    _check_classes(conn, pid)

    if body.image_ids is None:
        total = conn.execute("SELECT COUNT(*) FROM images WHERE project_id=?",
                             (pid,)).fetchone()[0]
    else:
        ids = sorted(set(body.image_ids))
        found = set()
        for start in range(0, len(ids), 500):
            chunk = ids[start:start + 500]
            marks = ",".join("?" * len(chunk))
            found.update(r[0] for r in conn.execute(
                f"SELECT id FROM images WHERE project_id=? AND id IN ({marks})", [pid] + chunk))
        missing = [i for i in ids if i not in found]
        if missing:
            raise ApiError(400, "bad_image_ids",
                           f"images not in project {pid}: {missing[:20]}")
        total = len(ids)

    config = {**body.model_dump(exclude={"rerun"}), "mode": mode}
    return _insert_job(request, conn, pid, total, config)


@router.get("/jobs/{jid}")
def get_job(jid: int, conn: sqlite3.Connection = Depends(get_conn)):
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if row is None:
        raise not_found("job", jid)
    return db.job_dict(row)
