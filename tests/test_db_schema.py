"""Tests for app/db/db.py: tables, foreign keys, stale jobs."""

import sqlite3

import pytest

from app.db import db


def test_init_creates_all_tables_and_is_repeatable(tmp_path):
    path = tmp_path / "x" / "app.db"
    db.init_db(path)
    db.init_db(path)  # second call must not fail
    conn = db.connect(path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"projects", "images", "classes", "jobs", "annotations"} <= names


def test_foreign_keys_on(tmp_path):
    db.init_db(tmp_path / "app.db")
    conn = db.connect(tmp_path / "app.db")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO images (project_id, file_name, path, width, height, uploaded_at)"
                     " VALUES (99, 'a.jpg', 'p', 1, 1, 't')")


def test_status_check_constraint(tmp_path):
    db.init_db(tmp_path / "app.db")
    conn = db.connect(tmp_path / "app.db")
    conn.execute("INSERT INTO projects (name, created_at) VALUES ('p', 't')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO jobs (project_id, kind, status, config, created_at) "
                     "VALUES (1, 'detect', 'weird', '{}', 't')")


def test_fail_stale_jobs(tmp_path):
    db.init_db(tmp_path / "app.db")
    conn = db.connect(tmp_path / "app.db")
    with conn:
        conn.execute("INSERT INTO projects (name, created_at) VALUES ('p', 't')")
        for status in ("running", "queued", "done"):
            conn.execute("INSERT INTO jobs (project_id, kind, status, config, created_at) "
                         "VALUES (1, 'detect', ?, '{}', 't')", (status,))
    assert db.fail_stale_jobs(conn) == 1
    statuses = [r[0] for r in conn.execute("SELECT status FROM jobs ORDER BY id")]
    assert statuses == ["failed", "queued", "done"]


def test_now_format():
    t = db.now()
    assert len(t) == 20 and t.endswith("Z") and t[10] == "T"
