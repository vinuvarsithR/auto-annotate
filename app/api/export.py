"""
export.py - endpoints 14 (COCO / YOLO / YOLO-seg zip download) and 21 (start a Roboflow push job).

Both only load rows and call autolabel.export; the rules (which images, which boxes,
zip layout) live there. #21 only checks and queues the job; the worker does the uploads.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.db import db
from autolabel.export import (
    FORMATS,
    NothingToExport,
    build_zip,
    roboflow_settings,
    validate_statuses,
)

router = APIRouter()


class ExportIn(BaseModel):
    format: str = "coco"
    statuses: list[str] | None = None
    include_images: bool = False


class PushIn(BaseModel):
    statuses: list[str] | None = None


def _statuses(statuses: list[str] | None) -> list[str]:
    try:
        return validate_statuses(statuses)
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "project"


@router.post("/projects/{pid}/export")
def export_project(pid: int, body: ExportIn | None = Body(default=None),
                   conn: sqlite3.Connection = Depends(get_conn)):
    project = require_project(conn, pid)
    body = body or ExportIn()
    if body.format not in FORMATS:
        raise ApiError(400, "bad_request", f"format must be one of {list(FORMATS)}")
    statuses = _statuses(body.statuses)

    image_rows = conn.execute("SELECT * FROM images WHERE project_id=? ORDER BY id",
                              (pid,)).fetchall()
    paths = {r["id"]: r["path"] for r in image_rows}
    images = [db.image_dict(r) for r in image_rows]
    classes = [db.class_dict(r) for r in conn.execute(
        "SELECT * FROM classes WHERE project_id=? ORDER BY id", (pid,))]
    annotations = [db.annotation_dict(r) for r in conn.execute(
        db.ANNOTATION_SELECT + " JOIN images i ON i.id = a.image_id "
        "WHERE i.project_id=? ORDER BY a.id", (pid,))]

    image_bytes = None
    if body.include_images:
        for im in images:
            if im["review_status"] in ("done", "empty") and not Path(paths[im["id"]]).is_file():
                raise ApiError(404, "not_found", f"file of image {im['id']} missing on disk")
        image_bytes = lambda im: Path(paths[im["id"]]).read_bytes()  # noqa: E731

    try:
        blob = build_zip(body.format, images, classes, annotations, statuses, image_bytes)
    except NothingToExport:
        raise ApiError(409, "conflict", f"project {pid} has no done or empty images to export")
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))

    fname = f"{_safe_filename(project['name'])}_{body.format}.zip"
    return Response(blob, media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.post("/projects/{pid}/roboflow/push", status_code=202)
def push_project(pid: int, request: Request, body: PushIn | None = Body(default=None),
                 conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    statuses = _statuses((body or PushIn()).statuses)

    settings = roboflow_settings()
    if isinstance(settings, list):   # names only, never values
        raise ApiError(409, "conflict",
                       "Roboflow settings missing from .env: " + ", ".join(settings))

    n_ready = conn.execute(
        "SELECT COUNT(*) FROM images WHERE project_id=? AND review_status IN ('done', 'empty')",
        (pid,)).fetchone()[0]
    if not n_ready:
        raise ApiError(409, "conflict", f"project {pid} has no done or empty images to push")
    total = conn.execute(
        "SELECT COUNT(*) FROM images WHERE project_id=? AND review_status IN ('done', 'empty') "
        "AND roboflow_image_id IS NULL", (pid,)).fetchone()[0]

    return db.create_job(conn, pid, "roboflow_push", {"statuses": statuses}, total,
                         wake=request.app.state.worker.wake)
