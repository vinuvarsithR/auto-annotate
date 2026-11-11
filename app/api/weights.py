"""
weights.py - endpoints 24-26: upload, list and delete trained weights (Train on Colab).

Plain-English summary
---------------------
* #24 saves the uploaded .pt in a staging file, checks it with inspect_weights (the models
  session's code, injected through create_app), and only when every check passes moves it to
  data/weights/<pid>/<name>.pt and adds the trained_models row. Any failure deletes the file.
* The name is <safe project name>_v<version>; version = 1 + the highest this project ever had
  (deleted rows included), so a number is never reused. Name used by another project ->
  <safe project name>_p<pid>_v<version>.
* #25 lists the non-deleted rows by version. #26 deletes the file, keeps the row with
  deleted_at set and calls unregister_trained(name); 409 while a queued/running detect job
  uses the model.
* Security: a .pt file is a pickle; loading it runs code. Upload only weights you trained.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError, not_found
from app.db import db

router = APIRouter()

MAX_WEIGHTS_BYTES = 200 * 1024 * 1024  # 200 MB
TASKS = ("detect", "segment")
PARTS = {"file", "task"}


def safe_name(name: str) -> str:
    """Project name made safe for file names, the same rule as #14's zip name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "project"


def _bad(message: str) -> ApiError:
    return ApiError(400, "bad_request", message)


def _check_classes(info, project_classes: list[str]) -> None:
    """Names must match as sets, exact case, no repeats (never remapped)."""
    model = [str(n) for n in info.class_names]
    seen, repeated = set(), []
    for n in model:
        if n in seen and n not in repeated:
            repeated.append(n)
        seen.add(n)
    missing = [n for n in project_classes if n not in seen]
    extra = list(dict.fromkeys([n for n in model if n not in project_classes] + repeated))
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing in model: " + ", ".join(missing))
        if extra:
            parts.append("extra in model: " + ", ".join(extra))
        raise ApiError(400, "class_mismatch",
                       "model classes do not match project classes: " + "; ".join(parts),
                       details={"model_classes": model, "project_classes": project_classes,
                                "missing_in_model": missing, "extra_in_model": extra})


def _save_upload(f: UploadFile, target: Path) -> int:
    """Copy the part to `target`; returns its size. Over the cap -> 400 (file removed)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    f.file.seek(0)
    with open(target, "wb") as out:
        while chunk := f.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_WEIGHTS_BYTES:
                break
            out.write(chunk)
    if size > MAX_WEIGHTS_BYTES:
        target.unlink(missing_ok=True)
        raise _bad(f"weights file is larger than {MAX_WEIGHTS_BYTES} bytes (200 MB)")
    return size


def _store(request: Request, conn: sqlite3.Connection, pid: int, f: UploadFile,
           task: str) -> dict:
    project = require_project(conn, pid)
    class_names = [r["name"] for r in conn.execute(
        "SELECT name FROM classes WHERE project_id=? ORDER BY id", (pid,))]
    if not class_names:
        raise ApiError(400, "no_classes", f"project {pid} has no classes; PUT the classes first")

    weights_root = Path(request.app.state.data_dir) / "weights"
    staging = weights_root / f"staging_{uuid.uuid4().hex}.pt"
    try:
        size = _save_upload(f, staging)
        if size == 0:
            raise _bad("the weights file is empty")
        try:
            info = request.app.state.inspect_weights(str(staging))
        except ValueError as exc:
            raise ApiError(400, "bad_weights",
                           f"{f.filename} is not a loadable ultralytics checkpoint ({exc})")
        if info.task not in TASKS:
            raise ApiError(400, "bad_weights",
                           f"the weights are a '{info.task}' model; only detect or segment work")
        if info.task != task:
            raise ApiError(400, "task_mismatch",
                           f"the weights are a '{info.task}' model, not '{task}'")
        _check_classes(info, class_names)

        base = safe_name(project["name"])
        conn.execute("BEGIN IMMEDIATE")
        target = None
        try:
            version = 1 + conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM trained_models WHERE project_id=?",
                (pid,)).fetchone()[0]
            name = f"{base}_v{version}"
            if conn.execute("SELECT 1 FROM trained_models WHERE name=?", (name,)).fetchone():
                name = f"{base}_p{pid}_v{version}"
            target = weights_root / str(pid) / f"{name}.pt"
            cur = conn.execute(
                "INSERT INTO trained_models (project_id, name, version, task, class_names, imgsz, "
                "file_name, path, size_bytes, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pid, name, version, info.task, json.dumps([str(n) for n in info.class_names]),
                 int(info.imgsz) if info.imgsz is not None else None,
                 os.path.basename((f.filename or "").replace("\\", "/")), str(target), size,
                 db.now()))
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, target)
            conn.commit()
        except BaseException:
            conn.rollback()
            if target is not None:
                target.unlink(missing_ok=True)
            raise
        return db.trained_model_dict(conn.execute(
            "SELECT * FROM trained_models WHERE id=?", (cur.lastrowid,)).fetchone())
    finally:
        staging.unlink(missing_ok=True)


@router.post("/projects/{pid}/weights", status_code=201)
async def upload_weights(pid: int, request: Request,
                         conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)  # 404 before looking at the body
    try:
        form = await request.form(max_part_size=MAX_WEIGHTS_BYTES + 1)
    except Exception as exc:
        raise _bad(f"could not read the multipart form: {exc}")
    try:
        keys = list(form.keys())
        extra = sorted(set(keys) - PARTS)
        if extra:
            raise _bad(f"unexpected form parts: {', '.join(extra)}")
        files, tasks = form.getlist("file"), form.getlist("task")
        if len(files) != 1 or not isinstance(files[0], UploadFile):
            raise _bad("send exactly one 'file' part (the .pt weights)")
        if len(tasks) != 1 or not isinstance(tasks[0], str) or tasks[0] not in TASKS:
            raise _bad("form field 'task' is required: detect or segment")
        f = files[0]
        if not (f.filename or "").lower().endswith(".pt"):
            raise _bad("the weights file name must end with .pt")
        return await run_in_threadpool(_store, request, conn, pid, f, tasks[0])
    finally:
        await form.close()


@router.get("/projects/{pid}/weights")
def list_weights(pid: int, conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    rows = conn.execute("SELECT * FROM trained_models WHERE project_id=? AND deleted_at IS NULL "
                        "ORDER BY version", (pid,)).fetchall()
    return {"weights": [db.trained_model_dict(r) for r in rows]}


def model_in_use(conn: sqlite3.Connection, pid: int, name: str) -> bool:
    """A queued or running detect job of the project whose config.model is `name`."""
    for (config,) in conn.execute("SELECT config FROM jobs WHERE project_id=? AND kind='detect' "
                                  "AND status IN ('queued', 'running')", (pid,)):
        try:
            if json.loads(config or "{}").get("model") == name:
                return True
        except (ValueError, AttributeError):
            continue
    return False


@router.delete("/weights/{wid}", status_code=204)
def delete_weights(wid: int, request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    conn.execute("BEGIN IMMEDIATE")  # no job can be queued between the check and the delete
    try:
        row = conn.execute("SELECT * FROM trained_models WHERE id=? AND deleted_at IS NULL",
                           (wid,)).fetchone()
        if row is None:
            raise not_found("weights", wid)
        if model_in_use(conn, row["project_id"], row["name"]):
            raise ApiError(409, "conflict",
                           f"model {row['name']} is used by a queued or running job")
        conn.execute("UPDATE trained_models SET deleted_at=? WHERE id=?", (db.now(), wid))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    Path(row["path"]).unlink(missing_ok=True)
    request.app.state.unregister_trained(row["name"])
    return Response(status_code=204)


def remove_project_weights(request: Request, pid: int, names: list[str]) -> None:
    """#18: after the project's rows are gone, its weights folder and registrations go too."""
    shutil.rmtree(Path(request.app.state.data_dir) / "weights" / str(pid), ignore_errors=True)
    for name in names:
        request.app.state.unregister_trained(name)
