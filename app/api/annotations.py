"""
annotations.py - endpoints 11-13 (read boxes, add a human box, review / edit a box)
and 19 (hard-delete a box).

Status rules for PATCH (docs/CONTRACT.md section 1 and #13):
| PATCH body                                          | new status  |
| explicit status (any of the 5)                      | that status |
| no status; bbox / mask / class_id actually changed  | edited      |
| no status; nothing changed (e.g. only reviewer)     | unchanged   |
A "rejected" row is kept. Only DELETE (#19) removes a row.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field, field_validator

from app.api.deps import get_conn, require_image
from app.api.errors import ApiError, not_found
from app.db import db

router = APIRouter()


def _check_bbox(v):
    if v is not None and (v[2] <= 0 or v[3] <= 0):
        raise ValueError("bbox width and height must be > 0")
    return v


class AnnotationIn(BaseModel):
    class_id: int
    bbox: list[float] = Field(min_length=4, max_length=4)
    mask: dict | None = None
    reviewer: str | None = None

    _bbox = field_validator("bbox")(_check_bbox)


class AnnotationPatch(BaseModel):
    status: Literal["auto_accepted", "needs_review", "accepted", "rejected", "edited"] | None = None
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    mask: dict | None = None
    class_id: int | None = None
    reviewer: str | None = None

    _bbox = field_validator("bbox")(_check_bbox)


def _require_class_in_project(conn: sqlite3.Connection, class_id: int, pid: int) -> None:
    if not conn.execute("SELECT 1 FROM classes WHERE id=? AND project_id=?",
                        (class_id, pid)).fetchone():
        raise ApiError(400, "bad_class_id", f"class {class_id} is not in project {pid}")


@router.get("/images/{iid}/annotations")
def list_annotations(iid: int, status: str | None = None,
                     conn: sqlite3.Connection = Depends(get_conn)):
    image = require_image(conn, iid)
    if status is not None and status not in db.STATUSES:
        raise ApiError(400, "bad_request", f"unknown status '{status}'")
    sql, args = db.ANNOTATION_SELECT + " WHERE a.image_id=?", [iid]
    if status is not None:
        sql += " AND a.status=?"
        args.append(status)
    rows = conn.execute(sql + " ORDER BY a.id", args).fetchall()
    counts = db.image_counts(conn, [iid])[iid]
    return {"image": db.image_dict(image, counts),
            "annotations": [db.annotation_dict(r) for r in rows]}


@router.post("/images/{iid}/annotations", status_code=201)
def add_annotation(iid: int, body: AnnotationIn, conn: sqlite3.Connection = Depends(get_conn)):
    image = require_image(conn, iid)
    _require_class_in_project(conn, body.class_id, image["project_id"])
    x, y, w, h = body.bbox
    t = db.now()
    conn.execute("BEGIN IMMEDIATE")  # #20 cannot mark the image empty between check and insert
    with conn:
        if conn.execute("SELECT review_status FROM images WHERE id=?",
                        (iid,)).fetchone()[0] == "empty":
            raise ApiError(409, "conflict",
                           f"image {iid} is marked empty; set review_status to todo first")
        cur = conn.execute(
            "INSERT INTO annotations (image_id, class_id, x, y, w, h, mask, score, source, status, "
            "job_id, reviewer, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'human', 'accepted', NULL, ?, ?, ?)",
            (iid, body.class_id, x, y, w, h,
             json.dumps(body.mask) if body.mask is not None else None, body.reviewer, t, t),
        )
    return db.annotation_dict(db.get_annotation(conn, cur.lastrowid))


@router.patch("/annotations/{aid}")
def patch_annotation(aid: int, body: AnnotationPatch,
                     conn: sqlite3.Connection = Depends(get_conn)):
    row = db.get_annotation(conn, aid)
    if row is None:
        raise not_found("annotation", aid)
    sent = body.model_fields_set
    old = db.annotation_dict(row)

    updates: dict = {}
    changed = False
    if "bbox" in sent and body.bbox is not None:
        if [float(v) for v in body.bbox] != old["bbox"]:
            updates.update(zip("xywh", body.bbox))
            changed = True
    if "mask" in sent and body.mask != old["mask"]:
        updates["mask"] = json.dumps(body.mask) if body.mask is not None else None
        changed = True
    if "class_id" in sent and body.class_id is not None and body.class_id != old["class_id"]:
        image = require_image(conn, row["image_id"])
        _require_class_in_project(conn, body.class_id, image["project_id"])
        updates["class_id"] = body.class_id
        changed = True
    if "reviewer" in sent:
        updates["reviewer"] = body.reviewer

    if body.status is not None:
        updates["status"] = body.status  # explicit status wins; undo relies on this
    elif changed:
        updates["status"] = "edited"

    if updates:
        updates["updated_at"] = db.now()
        cols = ", ".join(f"{k}=?" for k in updates)
        with conn:
            conn.execute(f"UPDATE annotations SET {cols} WHERE id=?", [*updates.values(), aid])
    return db.annotation_dict(db.get_annotation(conn, aid))


@router.delete("/annotations/{aid}", status_code=204)
def delete_annotation(aid: int, conn: sqlite3.Connection = Depends(get_conn)):
    with conn:
        cur = conn.execute("DELETE FROM annotations WHERE id=?", (aid,))
    if cur.rowcount == 0:
        raise not_found("annotation", aid)
    return Response(status_code=204)
