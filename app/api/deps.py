"""deps.py - shared FastAPI dependencies: one SQLite connection per request."""

from __future__ import annotations

import sqlite3
from typing import Iterator

from fastapi import Request

from app.db import db
from app.api.errors import not_found


def get_conn(request: Request) -> Iterator[sqlite3.Connection]:
    conn = db.connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


def require_project(conn: sqlite3.Connection, pid: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    if row is None:
        raise not_found("project", pid)
    return row


def require_image(conn: sqlite3.Connection, iid: int) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM images WHERE id=?", (iid,)).fetchone()
    if row is None:
        raise not_found("image", iid)
    return row
