"""projects.py - endpoints 1-3 (create, list, get), 17 (rename) and 18 (delete)."""

from __future__ import annotations

import logging
import shutil
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, Request, Response
from pydantic import BaseModel, Field

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.db import db

router = APIRouter()
log = logging.getLogger("app")


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


@router.patch("/projects/{pid}")
def rename_project(pid: int, body: ProjectIn, conn: sqlite3.Connection = Depends(get_conn)):
    row = require_project(conn, pid)
    name = body.name.strip()
    if not name:
        raise ApiError(400, "bad_request", "name must not be empty")
    if name != row["name"]:
        try:
            with conn:
                conn.execute("UPDATE projects SET name=? WHERE id=?", (name, pid))
        except sqlite3.IntegrityError:
            raise ApiError(409, "name_taken", f"project name '{name}' is already used")
    return db.project_dict(conn, require_project(conn, pid))


@router.delete("/projects/{pid}", status_code=204)
def delete_project(pid: int, request: Request, conn: sqlite3.Connection = Depends(get_conn)):
    """Delete the project, all its rows and data/uploads/<pid>/.

    BEGIN IMMEDIATE takes the write lock first, so no job can be queued or started
    between the "no active job" check and the deletes.
    annotations.class_id / job_id have no ON DELETE CASCADE, so children go first.
    """
    require_project(conn, pid)
    conn.execute("BEGIN IMMEDIATE")
    try:
        if conn.execute("SELECT 1 FROM jobs WHERE project_id=? AND status IN ('queued', 'running')",
                        (pid,)).fetchone():
            raise ApiError(409, "conflict", f"project {pid} has a running job")
        conn.execute("DELETE FROM annotations WHERE image_id IN "
                     "(SELECT id FROM images WHERE project_id=?)", (pid,))
        for table in ("jobs", "classes", "images"):
            conn.execute(f"DELETE FROM {table} WHERE project_id=?", (pid,))
        conn.execute("DELETE FROM projects WHERE id=?", (pid,))
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    folder = Path(request.app.state.data_dir) / "uploads" / str(pid)
    try:
        shutil.rmtree(folder)
    except FileNotFoundError:
        pass
    except OSError:
        # The rows are already gone, so report success but leave a trace for cleanup.
        log.warning("project %s deleted, but could not remove %s", pid, folder, exc_info=True)
    return Response(status_code=204)
