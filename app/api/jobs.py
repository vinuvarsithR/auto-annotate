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

    try:
        available = list(request.app.state.list_detectors())
    except ImportError:
        available = []
    if body.model not in available:
        msg = (f"unknown model '{body.model}', available: {', '.join(available)}" if available
               else f"model '{body.model}' is not available (autolabel.models is not installed)")
        raise ApiError(400, "unknown_model", msg)
    mode = _resolve_mode(request, body.model, body.mode)

    if not conn.execute("SELECT 1 FROM classes WHERE project_id=?", (pid,)).fetchone():
        raise ApiError(400, "no_classes", f"project {pid} has no classes; PUT the classes first")

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

    config = {**body.model_dump(), "mode": mode}
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


@router.get("/jobs/{jid}")
def get_job(jid: int, conn: sqlite3.Connection = Depends(get_conn)):
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
    if row is None:
        raise not_found("job", jid)
    return db.job_dict(row)
