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
* store_weights() is that storage step, shared by #24, #38 and (later) the local route.
  #24 may add a `metrics` part (<run>_metrics.json, read by export's read_metrics).
* #38 copies a .pt (and its _metrics.json) from an import folder (AUTOLABEL_IMPORT_DIRS,
  default data/inbox) on this host's disk; paths are resolved first, so .. and symlinks
  cannot leave the folder.
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
from typing import Any, Callable

from fastapi import APIRouter, Body, Depends, Request, Response
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError, not_found
from app.db import db

router = APIRouter()

MAX_WEIGHTS_BYTES = 200 * 1024 * 1024  # 200 MB
TASKS = ("detect", "segment")
PARTS = {"file", "task", "metrics"}
MAX_METRICS_BYTES = 1024 * 1024  # 1 MB


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


# ------------------------------------------------------------------ metrics file (#24, #38)

def read_metrics(data: bytes, task: str) -> dict:
    """export's read_metrics -> {"metrics", "split"}; its BadMetrics -> 400 bad_metrics."""
    from autolabel.export import train_package as tp  # export's module: imported, never edited

    if not 1 <= len(data) <= MAX_METRICS_BYTES:
        raise ApiError(400, "bad_metrics", "metrics file must be 1 byte .. 1 MB")
    try:
        return tp.read_metrics(data, task)
    except tp.BadMetrics as exc:
        raise ApiError(400, "bad_metrics", str(exc))


# ------------------------------------------------------------------ shared storage

def store_weights(conn: sqlite3.Connection, data_dir: str | Path, pid: int, src: Path,
                  file_name: str, task: str, source: str, metrics: dict | None,
                  inspect_weights: Callable, *, move: bool = False) -> dict:
    """#24 / #38 / (later) the local route: check `src` and register it as the project's next
    trained model. Same checks, order, naming and versioning as #24. `src` is copied; it is only
    moved when `move` is True (#24's staging file, which the caller deletes on failure).
    `metrics` is read_metrics' {"metrics", "split"} or None. Returns the TrainedModel."""
    src = Path(src)
    project = require_project(conn, pid)
    class_names = [r["name"] for r in conn.execute(
        "SELECT name FROM classes WHERE project_id=? ORDER BY id", (pid,))]
    if not class_names:
        raise ApiError(400, "no_classes", f"project {pid} has no classes; PUT the classes first")
    size = src.stat().st_size
    if size == 0:
        raise _bad("the weights file is empty")
    if size > MAX_WEIGHTS_BYTES:
        raise _bad(f"weights file is larger than {MAX_WEIGHTS_BYTES} bytes (200 MB)")
    try:
        info = inspect_weights(str(src))
    except ValueError as exc:
        raise ApiError(400, "bad_weights",
                       f"{file_name} is not a loadable ultralytics checkpoint ({exc})")
    if info.task not in TASKS:
        raise ApiError(400, "bad_weights",
                       f"the weights are a '{info.task}' model; only detect or segment work")
    if info.task != task:
        raise ApiError(400, "task_mismatch",
                       f"the weights are a '{info.task}' model, not '{task}'")
    _check_classes(info, class_names)

    weights_root = Path(data_dir) / "weights"
    base = safe_name(project["name"])
    metrics = metrics or {}
    m_json = json.dumps(metrics["metrics"]) if metrics.get("metrics") is not None else None
    s_json = json.dumps(metrics["split"]) if metrics.get("split") is not None else None
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
            "file_name, path, size_bytes, uploaded_at, source, metrics, split) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pid, name, version, info.task, json.dumps([str(n) for n in info.class_names]),
             int(info.imgsz) if info.imgsz is not None else None,
             os.path.basename((file_name or "").replace("\\", "/")), str(target), size,
             db.now(), source, m_json, s_json))
        target.parent.mkdir(parents=True, exist_ok=True)
        if move:
            os.replace(src, target)
        else:  # copy under a temp name first, so a half-copied file never has the final name
            part = target.with_suffix(".part")
            shutil.copyfile(src, part)
            os.replace(part, target)
        conn.commit()
    except BaseException:
        conn.rollback()
        if target is not None:
            target.unlink(missing_ok=True)
            target.with_suffix(".part").unlink(missing_ok=True)
        raise
    return db.trained_model_dict(conn.execute(
        "SELECT * FROM trained_models WHERE id=?", (cur.lastrowid,)).fetchone())


def _store(request: Request, conn: sqlite3.Connection, pid: int, f: UploadFile,
           task: str, metrics_part: UploadFile | None) -> dict:
    require_project(conn, pid)
    metrics = None
    if metrics_part is not None:  # after the form's task check, before loading the weights
        metrics_part.file.seek(0)
        metrics = read_metrics(metrics_part.file.read(MAX_METRICS_BYTES + 1), task)
    staging = Path(request.app.state.data_dir) / "weights" / f"staging_{uuid.uuid4().hex}.pt"
    try:
        _save_upload(f, staging)
        return store_weights(conn, request.app.state.data_dir, pid, staging, f.filename or "",
                             task, "colab", metrics, request.app.state.inspect_weights,
                             move=True)
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
        metrics_parts = form.getlist("metrics")
        if len(files) != 1 or not isinstance(files[0], UploadFile):
            raise _bad("send exactly one 'file' part (the .pt weights)")
        if len(tasks) != 1 or not isinstance(tasks[0], str) or tasks[0] not in TASKS:
            raise _bad("form field 'task' is required: detect or segment")
        f = files[0]
        if not (f.filename or "").lower().endswith(".pt"):
            raise _bad("the weights file name must end with .pt")
        metrics_part = None
        if metrics_parts:
            if len(metrics_parts) != 1 or not isinstance(metrics_parts[0], UploadFile):
                raise _bad("send at most one 'metrics' part (the _metrics.json file)")
            metrics_part = metrics_parts[0]
            if not (metrics_part.filename or "").lower().endswith(".json"):
                raise _bad("the metrics file name must end with .json")
        return await run_in_threadpool(_store, request, conn, pid, f, tasks[0], metrics_part)
    finally:
        await form.close()


# ------------------------------------------------------------------ #38 import from disk

IMPORT_FIELDS = {"path", "task", "metrics_path"}


def import_dirs(data_dir: str | Path) -> list[Path]:
    """AUTOLABEL_IMPORT_DIRS (absolute folders separated by ;), resolved; default
    <data dir>/inbox. Read on every call (the lifespan loads .env into the environment)."""
    raw = (os.environ.get("AUTOLABEL_IMPORT_DIRS") or "").strip()
    if not raw:
        return [(Path(data_dir) / "inbox").resolve()]
    return [Path(p.strip()).resolve() for p in raw.split(";")
            if p.strip() and Path(p.strip()).is_absolute()]


def _allowed(given: str, dirs: list[Path], suffix: str) -> Path:
    """The resolved path (symlinks followed). Outside every import folder (or not absolute)
    -> 400 path_not_allowed; not a file -> 404 not_found; wrong suffix -> 400 bad_request."""
    p = Path(given)
    resolved = p.resolve() if p.is_absolute() else None
    if resolved is None or not any(resolved.is_relative_to(d) for d in dirs):
        raise ApiError(400, "path_not_allowed",
                       f"{given} is not inside an import folder (AUTOLABEL_IMPORT_DIRS)")
    if not resolved.is_file():
        raise ApiError(404, "not_found", f"{given} is not a file")
    if resolved.suffix.lower() != suffix:
        raise _bad(f"{given} must end with {suffix}")
    return resolved


def plan_import(conn: sqlite3.Connection, data_dir: str | Path, pid: int, body: Any) -> dict:
    """#38 entry checks that do not load the weights: 404 project, body, path rules, metrics.
    Returns {"path", "task", "metrics_path", "metrics"} (paths resolved)."""
    require_project(conn, pid)
    if not isinstance(body, dict):
        raise _bad("body must be a JSON object")
    extra = sorted(set(body) - IMPORT_FIELDS)
    if extra:
        raise _bad(f"unknown field(s): {', '.join(extra)}")
    path, task, metrics_path = body.get("path"), body.get("task"), body.get("metrics_path")
    if not isinstance(path, str) or not path:
        raise _bad("field 'path' is required: the absolute path of the .pt file")
    if task not in TASKS:
        raise _bad("field 'task' is required: detect or segment")
    if metrics_path is not None and (not isinstance(metrics_path, str) or not metrics_path):
        raise _bad("field 'metrics_path' must be a path string or null")
    dirs = import_dirs(data_dir)
    src = _allowed(path, dirs, ".pt")
    size = src.stat().st_size
    if size == 0 or size > MAX_WEIGHTS_BYTES:
        raise _bad(f"{path} must be 1 byte .. 200 MB")
    mpath = None
    if metrics_path is not None:
        mpath = _allowed(metrics_path, dirs, ".json")
    elif src.name.lower().endswith("_best.pt"):
        sibling = src.with_name(src.name[:-len("_best.pt")] + "_metrics.json")
        if sibling.is_file():
            mpath = _allowed(str(sibling), dirs, ".json")
    metrics = None
    if mpath is not None:
        if mpath.stat().st_size > MAX_METRICS_BYTES:
            raise ApiError(400, "bad_metrics", "metrics file must be 1 byte .. 1 MB")
        metrics = read_metrics(mpath.read_bytes(), task)
    return {"path": src, "task": task, "metrics_path": mpath, "metrics": metrics}


def run_import(conn: sqlite3.Connection, state: Any, pid: int, plan: dict) -> dict:
    """Copy and register a planned import; the original file stays where it is."""
    return store_weights(conn, state.data_dir, pid, plan["path"], plan["path"].name,
                         plan["task"], "colab", plan["metrics"], state.inspect_weights)


@router.post("/projects/{pid}/weights/import", status_code=201)
def import_weights(pid: int, request: Request, body: Any = Body(default=None),
                   conn: sqlite3.Connection = Depends(get_conn)):
    plan = plan_import(conn, request.app.state.data_dir, pid, {} if body is None else body)
    return run_import(conn, request.app.state, pid, plan)


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
