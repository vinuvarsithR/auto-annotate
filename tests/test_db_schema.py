"""Tests for app/db/db.py: tables, foreign keys, stale jobs, migration."""

import re
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


# --- migration: databases made before images.review_status (#20) ------------

def _make_old_db(path):
    """A database with the schema as it was before review_status, holding one image."""
    old = re.sub(r"\n\s*review_status TEXT[^\n]*\n[^\n]*-- set only by #20", "",
                 db.SCHEMA.read_text(encoding="utf-8"))
    assert "review_status" not in old
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(old)
    conn.execute("INSERT INTO projects (name, created_at) VALUES ('old', 't')")
    conn.execute("INSERT INTO images (project_id, file_name, path, width, height, uploaded_at) "
                 "VALUES (1, 'a.jpg', 'p/a.jpg', 1280, 720, 't')")
    conn.commit()
    conn.close()


def test_migrate_adds_review_status_and_keeps_rows(tmp_path):
    path = tmp_path / "app.db"
    _make_old_db(path)
    db.init_db(path)
    db.init_db(path)  # second run must be a no-op
    conn = db.connect(path)
    row = conn.execute("SELECT * FROM images").fetchone()
    assert (row["file_name"], row["width"], row["review_status"]) == ("a.jpg", 1280, "todo")
    assert conn.execute("SELECT name FROM projects").fetchone()[0] == "old"
    with pytest.raises(sqlite3.IntegrityError):  # CHECK constraint came along
        conn.execute("UPDATE images SET review_status='weird'")


def test_new_db_review_status_default_and_check(tmp_path):
    db.init_db(tmp_path / "app.db")
    conn = db.connect(tmp_path / "app.db")
    conn.execute("INSERT INTO projects (name, created_at) VALUES ('p', 't')")
    conn.execute("INSERT INTO images (project_id, file_name, path, width, height, uploaded_at)"
                 " VALUES (1, 'a.jpg', 'p', 1, 1, 't')")
    assert conn.execute("SELECT review_status FROM images").fetchone()[0] == "todo"
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("UPDATE images SET review_status='weird'")


def test_old_db_app_starts_and_serves_review_status(tmp_path):
    from tests.test_api_helpers import make_client
    _make_old_db(tmp_path / "data" / "app.db")
    with make_client(tmp_path)[0] as c:
        imgs = c.get("/api/projects/1/images").json()["images"]
        assert [(i["file_name"], i["review_status"]) for i in imgs] == [("a.jpg", "todo")]
        assert c.patch(f"/api/images/{imgs[0]['id']}",
                       json={"review_status": "done"}).json()["review_status"] == "done"


# --- migration: databases made before images.roboflow_image_id (#21, B1) -----

def _make_pre_push_db(path):
    """A database with review_status but no roboflow_image_id, holding one reviewed image."""
    old = re.sub(r"\n\s*roboflow_image_id TEXT,[^\n]*", "", db.SCHEMA.read_text(encoding="utf-8"))
    assert "roboflow_image_id" not in old and "review_status" in old
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(old)
    conn.execute("INSERT INTO projects (name, created_at) VALUES ('old', 't')")
    conn.execute("INSERT INTO images (project_id, file_name, path, width, height, uploaded_at, "
                 "review_status) VALUES (1, 'a.jpg', 'p/a.jpg', 1280, 720, 't', 'done')")
    conn.commit()
    conn.close()


def test_migrate_adds_roboflow_image_id_and_keeps_rows(tmp_path):
    path = tmp_path / "app.db"
    _make_pre_push_db(path)
    db.init_db(path)
    db.init_db(path)  # second run must be a no-op
    conn = db.connect(path)
    row = conn.execute("SELECT * FROM images").fetchone()
    assert (row["file_name"], row["review_status"], row["roboflow_image_id"]) == \
        ("a.jpg", "done", None)
    conn.execute("UPDATE images SET roboflow_image_id='abc'")
    assert conn.execute("SELECT roboflow_image_id FROM images").fetchone()[0] == "abc"


def test_oldest_db_gets_both_columns(tmp_path):
    """A database from before #20 and #21 gets review_status AND roboflow_image_id."""
    path = tmp_path / "app.db"
    old = re.sub(r"\n\s*review_status TEXT[^\n]*\n[^\n]*-- set only by #20", "",
                 db.SCHEMA.read_text(encoding="utf-8"))
    old = re.sub(r"\n\s*roboflow_image_id TEXT,[^\n]*", "", old)
    assert "review_status" not in old and "roboflow_image_id" not in old
    conn = sqlite3.connect(path)
    conn.executescript(old)
    conn.execute("INSERT INTO projects (name, created_at) VALUES ('old', 't')")
    conn.execute("INSERT INTO images (project_id, file_name, path, width, height, uploaded_at) "
                 "VALUES (1, 'a.jpg', 'p', 1, 1, 't')")
    conn.commit()
    conn.close()
    db.init_db(path)
    row = db.connect(path).execute("SELECT * FROM images").fetchone()
    assert (row["file_name"], row["review_status"], row["roboflow_image_id"]) ==         ("a.jpg", "todo", None)


def test_roboflow_image_id_not_in_image_json(tmp_path):
    from tests.test_api_helpers import make_client, setup_project
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        img = c.get(f"/api/projects/{s['pid']}/images").json()["images"][0]
        assert "roboflow_image_id" not in img


# --- migration: databases made before images.sha256 (#4 duplicate check) ----

def _make_pre_hash_db(path, files):
    """A database without images.sha256. files: list of (file_name, path on disk)."""
    old = re.sub(r"\n\s*sha256 +TEXT,[^\n]*", "", db.SCHEMA.read_text(encoding="utf-8"))
    assert "sha256" not in old and "roboflow_image_id" in old
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(old)
    conn.execute("INSERT INTO projects (name, created_at) VALUES ('old', 't')")
    for name, p in files:
        conn.execute("INSERT INTO images (project_id, file_name, path, width, height, uploaded_at, "
                     "review_status, roboflow_image_id) VALUES (1, ?, ?, 4, 4, 't', 'done', 'rf1')",
                     (name, str(p)))
    conn.commit()
    conn.close()


def test_migrate_backfills_sha256_and_keeps_rows(tmp_path):
    import hashlib
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"AAAA")
    b.write_bytes(b"AAAA")  # an old duplicate: both rows must survive
    path = tmp_path / "app.db"
    _make_pre_hash_db(path, [("a.jpg", a), ("b.jpg", b), ("gone.jpg", tmp_path / "gone.jpg")])
    db.init_db(path)
    db.init_db(path)  # second run must be a no-op
    conn = db.connect(path)
    rows = conn.execute("SELECT * FROM images ORDER BY id").fetchall()
    want = hashlib.sha256(b"AAAA").hexdigest()
    assert [(r["file_name"], r["sha256"]) for r in rows] == \
        [("a.jpg", want), ("b.jpg", want), ("gone.jpg", None)]
    assert all((r["review_status"], r["roboflow_image_id"], r["width"]) == ("done", "rf1", 4)
               for r in rows)


def test_backfill_retries_file_that_appears_later(tmp_path):
    late = tmp_path / "late.jpg"
    path = tmp_path / "app.db"
    _make_pre_hash_db(path, [("late.jpg", late)])
    db.init_db(path)
    late.write_bytes(b"later")
    db.init_db(path)
    assert db.connect(path).execute("SELECT sha256 FROM images").fetchone()[0] is not None


def test_old_db_upload_of_existing_file_is_duplicate(tmp_path):
    """After backfill, re-uploading an image the old database already had is skipped."""
    from tests.test_api_helpers import image_bytes, make_client
    data = image_bytes(colour=(7, 7, 7))
    disk = tmp_path / "old.jpg"
    disk.write_bytes(data)
    _make_pre_hash_db(tmp_path / "data" / "app.db", [("old.jpg", disk)])
    with make_client(tmp_path)[0] as c:
        body = c.post("/api/projects/1/images",
                      files=[("files", ("renamed_copy.jpg", data, "image/jpeg")),
                             ("files", ("old.jpg", image_bytes(), "image/jpeg"))]).json()
        assert body["skipped"] == [{"file_name": "renamed_copy.jpg", "reason": "duplicate"}]
        assert body["renamed"] == [{"from": "old.jpg", "to": "old_2.jpg"}]
