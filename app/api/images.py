"""
images.py - endpoints 4-6: upload, list and download images.

Upload rules:
* Only real JPEG / PNG files (checked by opening them with PIL, not just the name).
* A file name already in the project is skipped as "duplicate".
* Only the base name is kept, so "../../x.jpg" cannot write outside data/uploads.
"""

from __future__ import annotations

import io
import os
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.api.deps import get_conn, require_image, require_project
from app.api.errors import ApiError
from app.db import db

router = APIRouter()

_FORMATS = {"JPEG": {".jpg", ".jpeg"}, "PNG": {".png"}}
_MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}


def safe_name(raw: str | None) -> str:
    """Base name only, for both / and \\ separators."""
    return os.path.basename((raw or "").replace("\\", "/")).strip()


def read_image_size(data: bytes, name: str) -> tuple[int, int] | None:
    """(width, height) if the bytes are a JPEG/PNG matching the extension, else None."""
    ext = Path(name).suffix.lower()
    try:
        with Image.open(io.BytesIO(data)) as im:
            fmt, size = im.format, im.size
            im.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        return None
    if ext not in _FORMATS.get(fmt, set()):
        return None
    return size


@router.post("/projects/{pid}/images", status_code=201)
def upload_images(pid: int, request: Request, files: list[UploadFile] = File(...),
                  conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    folder = Path(request.app.state.data_dir) / "uploads" / str(pid)
    folder.mkdir(parents=True, exist_ok=True)

    added, skipped = [], []
    for f in files:
        name = safe_name(f.filename)
        data = f.file.read()
        size = read_image_size(data, name) if name else None
        if size is None:
            skipped.append({"file_name": name or (f.filename or ""), "reason": "not an image"})
            continue
        if conn.execute("SELECT 1 FROM images WHERE project_id=? AND file_name=?",
                        (pid, name)).fetchone():
            skipped.append({"file_name": name, "reason": "duplicate"})
            continue

        target = None
        try:
            with conn:
                cur = conn.execute(
                    "INSERT INTO images (project_id, file_name, path, width, height, uploaded_at) "
                    "VALUES (?, ?, '', ?, ?, ?)",
                    (pid, name, size[0], size[1], db.now()),
                )
                iid = cur.lastrowid
                target = folder / f"{iid}_{name}"
                target.write_bytes(data)
                conn.execute("UPDATE images SET path=? WHERE id=?", (target.as_posix(), iid))
        except Exception:
            if target is not None and target.exists():
                target.unlink()
            raise
        added.append(require_image(conn, iid))

    return {"images": db.images_dicts(conn, added), "skipped": skipped}


@router.get("/projects/{pid}/images")
def list_images(pid: int, status: str | None = None,
                offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=1000),
                conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    if status is not None and status not in db.STATUSES:
        raise ApiError(400, "bad_request", f"unknown status '{status}'")
    where = "WHERE i.project_id=?"
    args: list = [pid]
    if status is not None:
        where += (" AND EXISTS (SELECT 1 FROM annotations a "
                  "WHERE a.image_id = i.id AND a.status = ?)")
        args.append(status)
    total = conn.execute(f"SELECT COUNT(*) FROM images i {where}", args).fetchone()[0]
    rows = conn.execute(f"SELECT i.* FROM images i {where} ORDER BY i.id LIMIT ? OFFSET ?",
                        args + [limit, offset]).fetchall()
    return {"images": db.images_dicts(conn, rows), "total": total,
            "offset": offset, "limit": limit}


@router.get("/images/{iid}/file")
def image_file(iid: int, conn: sqlite3.Connection = Depends(get_conn)):
    row = require_image(conn, iid)
    path = Path(row["path"])
    if not path.is_file():
        raise ApiError(404, "not_found", f"file for image {iid} is missing on disk")
    media = _MEDIA.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media)
