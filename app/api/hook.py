"""
hook.py - endpoints 27-30: hook point per video + hook filter settings (OWLv2 + SAM 2).

* A "video" is sequence_of(images.file_name) (autolabel/sequences.py), same key as the #23 split.
* One hook point per (project, video), set by clicking an image of that video (#28).
* Settings live on the projects row: load_classes (JSON list) and hook_radius; NULL = default.
* The worker reads all of this once at the start of an owlv2 examples job.
* Bodies are read as plain JSON and checked here, so an unknown project is 404 before any 400.
"""

from __future__ import annotations

import json
import math
import sqlite3
from typing import Any

from fastapi import APIRouter, Body, Depends, Response

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.db import db

router = APIRouter()


def _sequence_of(file_name: str) -> str:
    from autolabel.sequences import sequence_of  # pure function, no torch
    return sequence_of(file_name)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _object(body: Any, allowed: set[str], required: set[str] = frozenset()) -> dict:
    if not isinstance(body, dict):
        raise ApiError(400, "bad_request", "body must be a JSON object")
    extra = sorted(set(body) - allowed)
    if extra:
        raise ApiError(400, "bad_request", f"unknown fields: {', '.join(extra)}")
    missing = sorted(required - set(body))
    if missing:
        raise ApiError(400, "bad_request", f"missing fields: {', '.join(missing)}")
    return body


def hook_settings_response(conn: sqlite3.Connection, pid: int) -> dict:
    """#27 HookSettings."""
    out = db.hook_settings(conn, pid)
    points = [db.hook_point_dict(r) for r in conn.execute(
        "SELECT * FROM hook_points WHERE project_id=? ORDER BY sequence", (pid,))]
    have = {p["sequence"] for p in points}
    counts: dict[str, int] = {}
    for (name,) in conn.execute("SELECT file_name FROM images WHERE project_id=?", (pid,)):
        seq = _sequence_of(name)
        counts[seq] = counts.get(seq, 0) + 1
    out["points"] = points
    out["sequences"] = [{"sequence": s, "images": counts[s], "has_point": s in have}
                        for s in sorted(counts)]
    return out


@router.get("/projects/{pid}/hook")
def get_hook(pid: int, conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    return hook_settings_response(conn, pid)


@router.put("/projects/{pid}/hook/points")
def put_hook_point(pid: int, body: Any = Body(None),
                   conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    body = _object(body, {"image_id", "x", "y"}, {"image_id", "x", "y"})
    iid, x, y = body["image_id"], body["x"], body["y"]
    if not isinstance(iid, int) or isinstance(iid, bool):
        raise ApiError(400, "bad_request", "image_id must be an integer")
    if not (_is_number(x) and _is_number(y)):
        raise ApiError(400, "bad_request", "x and y must be finite numbers")
    image = conn.execute("SELECT * FROM images WHERE id=? AND project_id=?",
                         (iid, pid)).fetchone()
    if image is None:
        raise ApiError(400, "bad_image_ids", f"images not in project {pid}: [{iid}]")
    w, h = image["width"], image["height"]
    if not (0 <= x <= w and 0 <= y <= h):
        raise ApiError(400, "bad_request", f"x must be in 0..{w} and y in 0..{h}")
    seq = _sequence_of(image["file_name"])
    with conn:
        conn.execute(
            "INSERT INTO hook_points (project_id, sequence, x, y, image_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (project_id, sequence) DO UPDATE SET "
            "x=excluded.x, y=excluded.y, image_id=excluded.image_id, "
            "updated_at=excluded.updated_at",
            (pid, seq, float(x), float(y), iid, db.now()))
    return db.hook_point_dict(conn.execute(
        "SELECT * FROM hook_points WHERE project_id=? AND sequence=?", (pid, seq)).fetchone())


@router.delete("/projects/{pid}/hook/points/{sequence}", status_code=204)
def delete_hook_point(pid: int, sequence: str, conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    with conn:
        cur = conn.execute("DELETE FROM hook_points WHERE project_id=? AND sequence=?",
                           (pid, sequence))
    if cur.rowcount == 0:
        raise ApiError(404, "not_found", f"no hook point for sequence {sequence}")
    return Response(status_code=204)


@router.patch("/projects/{pid}/hook")
def patch_hook(pid: int, body: Any = Body(None), conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    body = _object(body, {"load_classes", "radius"})
    updates: dict[str, Any] = {}
    if "load_classes" in body:
        names = body["load_classes"]
        if names is None:
            updates["load_classes"] = None
        else:
            if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
                raise ApiError(400, "bad_request", "load_classes must be a list of class names")
            repeated = sorted({n for n in names if names.count(n) > 1})
            if repeated:
                raise ApiError(400, "bad_request", f"repeated class names: {', '.join(repeated)}")
            known = {r[0] for r in conn.execute("SELECT name FROM classes WHERE project_id=?",
                                                (pid,))}
            unknown = [n for n in names if n not in known]
            if unknown:
                raise ApiError(400, "bad_request",
                               f"not classes of project {pid}: {', '.join(unknown)}")
            updates["load_classes"] = json.dumps(names)
    if "radius" in body:
        r = body["radius"]
        if r is not None and not (_is_number(r) and r > 0):
            raise ApiError(400, "bad_request", "radius must be a finite number > 0")
        updates["hook_radius"] = None if r is None else float(r)
    if updates:
        with conn:
            conn.execute(f"UPDATE projects SET {', '.join(f'{k}=?' for k in updates)} WHERE id=?",
                         (*updates.values(), pid))
    return hook_settings_response(conn, pid)
