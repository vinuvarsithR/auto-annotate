"""
classes.py - endpoints 7-8: list classes and replace the whole class list.

PUT matches classes by name: existing names are updated, new names are added,
missing names are deleted. A missing class that still has annotations (any
status, rejected ones too) blocks the whole request with 409 class_in_use.
"""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.db import db

router = APIRouter()


class ClassIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1)
    min_score: float | None = Field(default=None, ge=0, le=1)
    accept_score: float | None = Field(default=None, ge=0, le=1)


class ClassesIn(BaseModel):
    classes: list[ClassIn]


def _list(conn: sqlite3.Connection, pid: int) -> dict:
    rows = conn.execute("SELECT * FROM classes WHERE project_id=? ORDER BY id", (pid,))
    return {"classes": [db.class_dict(r) for r in rows]}


@router.get("/projects/{pid}/classes")
def list_classes(pid: int, conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    return _list(conn, pid)


@router.put("/projects/{pid}/classes")
def replace_classes(pid: int, body: ClassesIn, conn: sqlite3.Connection = Depends(get_conn)):
    require_project(conn, pid)
    wanted = [c.model_copy(update={"name": c.name.strip(), "description": c.description.strip()})
              for c in body.classes]
    names = [c.name for c in wanted]
    if any(not n for n in names):
        raise ApiError(400, "bad_request", "class name must not be empty")
    if len(set(names)) != len(names):
        raise ApiError(400, "bad_request", "class names must be unique")

    existing = {r["name"]: r["id"] for r in
                conn.execute("SELECT id, name FROM classes WHERE project_id=?", (pid,))}
    removed = [n for n in existing if n not in set(names)]
    in_use = [n for n in removed if conn.execute(
        "SELECT 1 FROM annotations WHERE class_id=? LIMIT 1", (existing[n],)).fetchone()]
    if in_use:
        raise ApiError(409, "class_in_use",
                       f"class(es) {', '.join(sorted(in_use))} still have annotations")

    with conn:
        for n in removed:
            conn.execute("DELETE FROM classes WHERE id=?", (existing[n],))
        for c in wanted:
            if c.name in existing:
                conn.execute("UPDATE classes SET description=?, min_score=?, accept_score=? "
                             "WHERE id=?",
                             (c.description, c.min_score, c.accept_score, existing[c.name]))
            else:
                conn.execute("INSERT INTO classes (project_id, name, description, min_score, "
                             "accept_score) VALUES (?, ?, ?, ?, ?)",
                             (pid, c.name, c.description, c.min_score, c.accept_score))
    return _list(conn, pid)
