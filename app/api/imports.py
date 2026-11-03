"""
imports.py - endpoint 22: start a bulk import job (one .zip, or a folder upload).

Plain-English summary
---------------------
* Checks happen here, before the job exists (CONTRACT #22): unknown project -> 404;
  no files, a bad zip, an unsafe or repeated path, or too many bytes -> 400 bad_request.
  The checks themselves are the export session's code (autolabel.importers.check_zip /
  check_names), loaded through app.state.worker.load_importers (the worker uses the same),
  so tests can pass a stub.
* The upload is first saved in data/imports/staging_<random>/, then moved to
  data/imports/<job_id>/ inside the transaction that creates the job, so the worker can
  never pick the job up before its files are in place.
    zip:    data/imports/<job_id>/upload.zip     (the worker extracts it to files/)
    folder: data/imports/<job_id>/files/<relative path of each part>
* The worker (app/worker/worker.py) does the rest and deletes the folder at the end.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import uuid
from pathlib import Path, PurePosixPath

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.db import db

router = APIRouter()

DEFAULT_MAX_GB = 5.0


def max_import_bytes() -> int:
    """AUTOLABEL_IMPORT_MAX_GB (GB = 10^9 bytes), default 5."""
    raw = os.environ.get("AUTOLABEL_IMPORT_MAX_GB", "").strip()
    try:
        gb = float(raw) if raw else DEFAULT_MAX_GB
    except ValueError:
        gb = DEFAULT_MAX_GB
    return int(gb * 1e9)


def _bad(message: str) -> ApiError:
    return ApiError(400, "bad_request", message)


def _part_size(f: UploadFile) -> int:
    if f.size is not None:
        return int(f.size)
    f.file.seek(0, os.SEEK_END)
    n = f.file.tell()
    f.file.seek(0)
    return n


def _save_part(f: UploadFile, root: Path, rel: str) -> None:
    """Write one part at root/rel; refuses anything that would land outside root."""
    target = (root / PurePosixPath(rel.replace("\\", "/"))).resolve()
    if not target.is_relative_to(root.resolve()):
        raise _bad(f"unsafe path in upload: {rel}")
    target.parent.mkdir(parents=True, exist_ok=True)
    f.file.seek(0)
    with open(target, "wb") as out:
        shutil.copyfileobj(f.file, out, 1024 * 1024)


@router.post("/projects/{pid}/import", status_code=202)
def start_import(pid: int, request: Request,
                 files: list[UploadFile] | None = File(None),
                 conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    if not files:
        raise _bad("no files part in the upload")

    try:
        imp = request.app.state.worker.load_importers()
    except ImportError:
        raise ApiError(500, "internal_error", "autolabel.importers is not installed")
    max_bytes = max_import_bytes()

    imports_dir = Path(request.app.state.data_dir) / "imports"
    staging = imports_dir / f"staging_{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    try:
        names = [f.filename or "" for f in files]
        if len(files) == 1 and names[0].lower().endswith(".zip"):
            zip_path = staging / "upload.zip"
            _save_part(files[0], staging, "upload.zip")
            try:
                total = int(imp.check_zip(zip_path, max_bytes))
            except imp.ImportRejected as exc:
                raise _bad(str(exc))
            config = {"upload": "zip", "file_name": os.path.basename(names[0].replace("\\", "/"))}
        else:
            if any(not n.strip() for n in names):
                raise _bad("every file part needs a file name (its relative path)")
            try:
                imp.check_names(names)
            except imp.ImportRejected as exc:
                raise _bad(str(exc))
            size = sum(_part_size(f) for f in files)
            if size > max_bytes:
                raise _bad(f"upload is {size} bytes, above the {max_bytes} byte cap "
                           "(AUTOLABEL_IMPORT_MAX_GB)")
            for f, rel in zip(files, names):
                _save_part(f, staging / "files", rel)
            exts = {e.lower() for e in imp.IMAGE_EXTS}
            total = sum(1 for n in names if PurePosixPath(n.replace("\\", "/")).suffix.lower()
                        in exts)
            config = {"upload": "files", "count": len(files)}

        def move_into_place(jid: int) -> None:
            final = imports_dir / str(jid)
            if final.exists():  # left over from an old database with the same id
                shutil.rmtree(final)
            staging.rename(final)

        worker = getattr(request.app.state, "worker", None)
        return db.create_job(conn, pid, "import", config, total,
                             wake=worker.wake if worker is not None else None,
                             before_commit=move_into_place)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
