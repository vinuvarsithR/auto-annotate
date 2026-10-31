"""projects.py - endpoints 1-3: create, list and get projects."""

from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.db import db

router = APIRouter()


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


@router.post("/projects", status_code=201)
def create_project(body: ProjectIn, conn: sqlite3.Connection = Depends(get_conn)):
    name = body.name.strip()
    if not name:
        raise ApiError(400, "bad_request", "name must not be empty")
    try:
        with conn:
            cur = conn.execute("INSERT INTO projects (name, created_at) VALUES (?, ?)",
                               (name, db.now()))
    except sqlite3.IntegrityError:
        raise ApiError(409, "name_taken", f"project name '{name}' is already used")
    return db.project_dict(conn, require_project(conn, cur.lastrowid))


@router.get("/projects")
def list_projects(conn: sqlite3.Connection = Depends(get_conn)):
    rows = conn.execute("SELECT * FROM projects ORDER BY id").fetchall()
    return {"projects": [db.project_dict(conn, r) for r in rows]}


@router.get("/projects/{pid}")
def get_project(pid: int, conn: sqlite3.Connection = Depends(get_conn)):
    return db.project_dict(conn, require_project(conn, pid))
