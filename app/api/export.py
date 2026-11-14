"""
export.py - endpoints 14 (COCO / YOLO / YOLO-seg zip download), 21 (start a Roboflow push job)
and 23 (Colab training package zip).

All three only load rows and call autolabel.export; the rules (which images, which boxes,
zip layout) live there. #21 only checks and queues the job; the worker does the uploads.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import Response
from pydantic import BaseModel

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.db import db
from autolabel.export import (
    FORMATS,
    NoPolygons,
    NothingToExport,
    build_train_package,
    build_zip,
    check_train_options,
    roboflow_settings,
    safe_name,
    validate_statuses,
)
from autolabel.export.train_package import EPOCHS, IMGSZ

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


_safe_filename = safe_name


def _load_rows(conn: sqlite3.Connection, pid: int):
    """(images, classes, annotations as API JSON, image id -> file path)."""
    image_rows = conn.execute("SELECT * FROM images WHERE project_id=? ORDER BY id",
                              (pid,)).fetchall()
    paths = {r["id"]: r["path"] for r in image_rows}
    images = [db.image_dict(r) for r in image_rows]
    classes = [db.class_dict(r) for r in conn.execute(
        "SELECT * FROM classes WHERE project_id=? ORDER BY id", (pid,))]
    annotations = [db.annotation_dict(r) for r in conn.execute(
        db.ANNOTATION_SELECT + " JOIN images i ON i.id = a.image_id "
        "WHERE i.project_id=? ORDER BY a.id", (pid,))]
    return images, classes, annotations, paths


@router.post("/projects/{pid}/export")
def export_project(pid: int, body: ExportIn | None = Body(default=None),
                   conn: sqlite3.Connection = Depends(get_conn)):
    project = require_project(conn, pid)
    body = body or ExportIn()
    if body.format not in FORMATS:
        raise ApiError(400, "bad_request", f"format must be one of {list(FORMATS)}")
    statuses = _statuses(body.statuses)
    images, classes, annotations, paths = _load_rows(conn, pid)

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


class _FileMissing(Exception):
    pass


@router.post("/projects/{pid}/train-package")
def train_package(pid: int, body: Any = Body(default=None),
                  conn: sqlite3.Connection = Depends(get_conn)):
    """#23: the Colab training package zip. The body is checked here (not by pydantic) so an
    unknown project answers 404 before a bad body answers 400, as the contract orders them."""
    project = require_project(conn, pid)
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ApiError(400, "bad_request", "body must be a JSON object")
    extra = sorted(set(body) - {"task", "epochs", "imgsz", "base_weights"})
    if extra:
        raise ApiError(400, "bad_request", f"unknown field(s): {', '.join(extra)}")
    task = body.get("task", "detect")
    epochs = body.get("epochs", EPOCHS)
    imgsz = body.get("imgsz", IMGSZ)
    try:
        base_weights = check_train_options(task, epochs, imgsz, body.get("base_weights"))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))

    images, classes, annotations, paths = _load_rows(conn, pid)

    def image_bytes(im: dict) -> bytes:
        path = Path(paths[im["id"]])
        if not path.is_file():
            raise _FileMissing(im["id"])
        return path.read_bytes()

    try:
        blob = build_train_package(project["name"], task, images, classes, annotations,
                                   image_bytes, epochs=epochs, imgsz=imgsz,
                                   base_weights=base_weights)
    except NothingToExport:
        raise ApiError(409, "conflict", f"project {pid} has no done or empty images to export")
    except NoPolygons:
        raise ApiError(409, "conflict", f"project {pid} has no exported box with a polygon")
    except _FileMissing as e:
        raise ApiError(404, "not_found", f"file of image {e.args[0]} missing on disk")
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))

    fname = f"{safe_name(project['name'])}_train_{task}.zip"
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
