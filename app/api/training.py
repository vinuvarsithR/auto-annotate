"""
training.py - endpoint 37: POST /projects/{pid}/training, one entry point for every training route.

Plain-English summary
---------------------
* Body: target (colab | local, required), task (default segment), epochs (1-500, default 50),
  imgsz (320-2048, multiple of 32, default 960), base_weights (yolo11{n,s,m,l,x}[-seg].pt; must
  match the task, default yolo11s-seg.pt / yolo11s.pt). Any other field -> 400 bad_request.
* target "local": this host has no CUDA GPU -> 503 no_gpu (message points to colab); with a GPU
  -> 501 not_implemented (the local route is reserved, not built).
* target "colab": the #23 training package with these values (export's route function, so the
  same checks and the same split), stored as a 24 h download (#35) like make_train_package.
* app.state.cuda_available() is injected through create_app (tests pass a fake).
"""

from __future__ import annotations

import io
import json
import re
import sqlite3
import zipfile
from typing import Any

from fastapi import APIRouter, Body, Depends, Request

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.assistant import downloads

router = APIRouter()

TARGETS = ("colab", "local")
TASKS = ("detect", "segment")
FIELDS = {"target", "task", "epochs", "imgsz", "base_weights"}
SIZES = ("n", "s", "m", "l", "x")
BASE_WEIGHTS = {"detect": tuple(f"yolo11{s}.pt" for s in SIZES),
                "segment": tuple(f"yolo11{s}-seg.pt" for s in SIZES)}
DEFAULTS = {"task": "segment", "epochs": 50, "imgsz": 960}
_FILENAME = re.compile(r'filename="([^"]+)"')


def _bad(message: str) -> ApiError:
    return ApiError(400, "bad_request", message)


def _int(body: dict, key: str, lo: int, hi: int) -> int:
    v = body.get(key, DEFAULTS[key])
    if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
        raise _bad(f"{key} must be a whole number {lo}-{hi}")
    return v


def check_body(body: Any) -> dict:
    """The #37 body with defaults filled in; anything wrong -> 400 bad_request."""
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise _bad("body must be a JSON object")
    extra = sorted(set(body) - FIELDS)
    if extra:
        raise _bad(f"unknown field(s): {', '.join(extra)}")
    target = body.get("target")
    if target not in TARGETS:
        raise _bad("field 'target' is required: colab or local")
    task = body.get("task", DEFAULTS["task"])
    if task not in TASKS:
        raise _bad("task must be detect or segment")
    epochs = _int(body, "epochs", 1, 500)
    imgsz = _int(body, "imgsz", 320, 2048)
    if imgsz % 32:
        raise _bad("imgsz must be a multiple of 32")
    base = body.get("base_weights", BASE_WEIGHTS[task][1])  # yolo11s by default
    if base not in BASE_WEIGHTS["detect"] + BASE_WEIGHTS["segment"]:
        raise _bad("base_weights must be one of " + ", ".join(BASE_WEIGHTS[task]))
    if base not in BASE_WEIGHTS[task]:
        raise _bad(f"base_weights {base} does not fit task {task}: use one of "
                   + ", ".join(BASE_WEIGHTS[task]))
    return {"target": target, "task": task, "epochs": epochs, "imgsz": imgsz,
            "base_weights": base}


def no_gpu_message(pid: int) -> str:
    return ("No CUDA GPU on this host, so local training is not possible. Use target \"colab\": "
            "it returns a training package for Google Colab; import the best.pt afterwards with "
            f"POST /projects/{pid}/weights or the import_weights tool.")


def check_target(state: Any, pid: int, values: dict) -> None:
    """The target errors of #37 (local only): 503 no_gpu or 501 not_implemented."""
    if values["target"] != "local":
        return
    if not state.cuda_available():
        raise ApiError(503, "no_gpu", no_gpu_message(pid))
    raise ApiError(501, "not_implemented",
                   "local GPU training is not built yet; use target \"colab\"")


def package_download(state: Any, response, fallback_name: str) -> dict:
    """Store a #23 zip response as a #35 download: {download_url, file_name, size_bytes,
    expires_at} plus the zip's split.json under "split"."""
    blob = bytes(response.body)
    m = _FILENAME.search(response.headers.get("content-disposition", ""))
    file_name = m.group(1) if m else fallback_name
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        split = json.loads(zf.read("split.json"))
    token, expires_at = downloads.save(state.data_dir, blob, file_name)
    return {"download_url": f"/api/assistant/downloads/{token}", "file_name": file_name,
            "size_bytes": len(blob), "expires_at": expires_at, "split": split}


def plan_split(conn: sqlite3.Connection, pid: int, task: str) -> dict | None:
    """The split.json (with `warnings`) of the package #23 would build now, without reading
    image files or building the zip (export's plan_train_package). The #23 409s as the route
    words them. None while export's plan_train_package is not merged yet (no warnings known)."""
    from app.api import export as export_api  # export's module: imported, never edited
    import autolabel.export as export_pkg
    from autolabel.export import train_package as tp

    plan = getattr(export_pkg, "plan_train_package", None) or getattr(tp, "plan_train_package",
                                                                      None)
    project = require_project(conn, pid)
    if plan is None:
        return None
    images, classes, annotations, _paths = export_api._load_rows(conn, pid)
    try:
        return plan(project["name"], task, images, classes, annotations)
    except export_pkg.NothingToExport:
        raise ApiError(409, "conflict", f"project {pid} has no done or empty images to export")
    except export_pkg.NoPolygons:
        raise ApiError(409, "conflict", f"project {pid} has no exported box with a polygon")


def colab_package(state: Any, conn: sqlite3.Connection, pid: int, values: dict) -> dict:
    """target colab: the #23 package with these values, stored for download; the #37 body."""
    from app.api import export as export_api  # export's module: imported, never edited
    from autolabel.export.train_package import safe_name

    project = require_project(conn, pid)
    task = values["task"]
    response = export_api.train_package(
        pid, {k: values[k] for k in ("task", "epochs", "imgsz", "base_weights")}, conn)
    pkg = package_download(state, response, f"project_{pid}_train_{task}.zip")
    split = pkg.pop("split")
    run = f"{safe_name(project['name'])}_{task}"
    return {**values, "package": pkg, "split": split, "warnings": split.get("warnings", []),
            "next_steps": [
        "Download the zip.",
        "Open train.ipynb in Google Colab (Runtime > Change runtime type > GPU).",
        f"Run the 3 code cells; it downloads {run}_best.pt and {run}_metrics.json.",
        "Import both: Train panel upload (#24) or the import_weights tool (#38)."]}


@router.post("/projects/{pid}/training", status_code=201)
def start_training(pid: int, request: Request, body: Any = Body(default=None),
                   conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)  # 404 first
    values = check_body(body)
    check_target(request.app.state, pid, values)
    return colab_package(request.app.state, conn, pid, values)
