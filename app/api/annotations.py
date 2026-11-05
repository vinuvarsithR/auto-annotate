"""
annotations.py - endpoints 11-13 (read boxes, add a human box, review / edit a box)
and 19 (hard-delete a box).

Status rules for PATCH (docs/CONTRACT.md section 1 and #13):
| PATCH body                                          | new status  |
| explicit status (any of the 5)                      | that status |
| no status; bbox / mask / polygon / class_id changed | edited      |
| no status; nothing changed (e.g. only reviewer)     | unchanged   |
A "rejected" row is kept. Only DELETE (#19) removes a row.

Polygons (#12, #13): points are normalized 0..1 and x, y, w, h stay filled in. When a row has
a polygon, its bbox is always the polygon's bounding box in pixels, computed here.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Literal

from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel, Field, field_validator, model_validator

from app.api.deps import get_conn, require_image
from app.api.errors import ApiError, not_found
from app.db import db

router = APIRouter()


def _check_bbox(v):
    if v is not None and (v[2] <= 0 or v[3] <= 0):
        raise ValueError("bbox width and height must be > 0")
    return v


MIN_AREA = 1e-12  # normalized units; below this the points are in a line (allows float noise)


def _is_coord(v) -> bool:
    # bool is an int in Python, so it is excluded by hand. The range check also rejects
    # NaN and infinity, and unlike math.isfinite it cannot overflow on a huge JSON integer.
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace formula. A shape that crosses itself can cancel out to 0."""
    pairs = zip(points, points[1:] + points[:1])
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in pairs)) / 2


def check_polygon(points: list | None) -> list[list[float]] | None:
    """The #12 rule: returns the points as floats, or raises ValueError. None is allowed."""
    if points is None:
        return None
    if len(points) < 3:
        raise ValueError("polygon needs at least 3 points")
    for p in points:
        if not (isinstance(p, list) and len(p) == 2 and all(_is_coord(v) for v in p)):
            raise ValueError("each polygon point must be 2 numbers in 0..1")
    out = [[float(x), float(y)] for x, y in points]
    if polygon_area(out) <= MIN_AREA:
        raise ValueError("polygon has zero area")
    return out


def bbox_of(points: list[list[float]], width: int, height: int) -> list[float]:
    """The polygon's bounding box [x, y, w, h] in pixels."""
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs) * width, min(ys) * height,
            (max(xs) - min(xs)) * width, (max(ys) - min(ys)) * height]


def fit_polygon(points: list[list[float]], bbox: list[float],
                width: int, height: int) -> list[list[float]]:
    """Scale and move the polygon to fill `bbox` (pixels), then clamp the points to 0..1."""
    x0, y0, w0, h0 = bbox_of(points, width, height)
    x1, y1, w1, h1 = bbox

    def place(v, full, old_start, old_size, new_start, new_size):
        moved = new_start + (v * full - old_start) / old_size * new_size
        return min(max(moved / full, 0.0), 1.0)

    return [[place(x, width, x0, w0, x1, w1), place(y, height, y0, h0, y1, h1)]
            for x, y in points]


def _ignore_bbox_with_polygon(data):
    """With a polygon the bbox is computed, so a sent bbox is dropped before it is validated."""
    if isinstance(data, dict) and data.get("polygon") is not None:
        return {k: v for k, v in data.items() if k != "bbox"}
    return data


class AnnotationIn(BaseModel):
    class_id: int
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    mask: dict | None = None
    polygon: list | None = None  # plain list: pydantic floats would accept true and "0.5"
    reviewer: str | None = None

    _drop_bbox = model_validator(mode="before")(_ignore_bbox_with_polygon)
    _bbox = field_validator("bbox")(_check_bbox)
    _polygon = field_validator("polygon")(check_polygon)

    @model_validator(mode="after")
    def _need_a_shape(self):
        if self.bbox is None and self.polygon is None:
            raise ValueError("bbox is required without a polygon")
        return self


class AnnotationPatch(BaseModel):
    status: Literal["auto_accepted", "needs_review", "accepted", "rejected", "edited"] | None = None
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    mask: dict | None = None
    polygon: list | None = None
    class_id: int | None = None
    reviewer: str | None = None

    _drop_bbox = model_validator(mode="before")(_ignore_bbox_with_polygon)
    _bbox = field_validator("bbox")(_check_bbox)
    _polygon = field_validator("polygon")(check_polygon)


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
    polygon = body.polygon
    x, y, w, h = (bbox_of(polygon, image["width"], image["height"]) if polygon is not None
                  else body.bbox)
    t = db.now()
    conn.execute("BEGIN IMMEDIATE")  # #20 cannot mark the image empty between check and insert
    with conn:
        if conn.execute("SELECT review_status FROM images WHERE id=?",
                        (iid,)).fetchone()[0] == "empty":
            raise ApiError(409, "conflict",
                           f"image {iid} is marked empty; set review_status to todo first")
        cur = conn.execute(
            "INSERT INTO annotations (image_id, class_id, x, y, w, h, mask, polygon, score, "
            "source, status, job_id, reviewer, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, 'human', 'accepted', NULL, ?, ?, ?)",
            (iid, body.class_id, x, y, w, h,
             json.dumps(body.mask) if body.mask is not None else None,
             json.dumps(polygon) if polygon is not None else None, body.reviewer, t, t),
        )
    return db.annotation_dict(db.get_annotation(conn, cur.lastrowid))


def _patch_shape(conn: sqlite3.Connection, row: sqlite3.Row, old: dict, body: AnnotationPatch,
                 updates: dict) -> bool:
    """Apply the bbox / polygon table of #13 to `updates`. Returns True if the shape changed."""
    sent = body.model_fields_set
    new_bbox = body.bbox if "bbox" in sent and body.bbox is not None else None
    if new_bbox is not None and [float(v) for v in new_bbox] == old["bbox"]:
        new_bbox = None
    image = require_image(conn, row["image_id"])
    width, height = image["width"], image["height"]

    if "polygon" in sent and body.polygon is not None:
        if body.polygon == old["polygon"]:
            return False
        polygon = body.polygon
    elif "polygon" not in sent and old["polygon"] is not None:
        if new_bbox is None:
            return False
        try:
            polygon = check_polygon(fit_polygon(old["polygon"], new_bbox, width, height))
        except ValueError:  # e.g. a bbox fully outside the image: all points clamp onto one edge
            raise ApiError(400, "bad_request",
                           "bbox leaves the polygon with no area inside the image")
    else:  # plain box before and after, or polygon: null
        removed = "polygon" in sent and old["polygon"] is not None
        if removed:
            updates["polygon"] = None
        if new_bbox is not None:
            updates.update(zip("xywh", new_bbox))
        return removed or new_bbox is not None

    updates["polygon"] = json.dumps(polygon)
    updates.update(zip("xywh", bbox_of(polygon, width, height)))
    return True


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
    if _patch_shape(conn, row, old, body, updates):
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
