"""#22 migration: the jobs table gains kind 'import' (table rebuild) and a result column.

The old database is built with the schema as it was before #22 (backend commit b886262),
filled with rows, then COPIED; only the copy is migrated, and every row of every table is
compared with the untouched original.
"""

import json
import shutil
import sqlite3

import pytest

from app.db import db

NEW_SCHEMA = db.SCHEMA.read_text(encoding="utf-8")
# schema.sql before #22: no 'import' kind, no result column
OLD_SCHEMA = (NEW_SCHEMA
              .replace("CHECK (kind IN ('detect', 'roboflow_push', 'import'))",
                       "CHECK (kind IN ('detect', 'roboflow_push'))")
              .replace("  result       TEXT,                      "
                       "-- JSON summary; set by import jobs (#22) and examples-mode detect jobs, NULL otherwise\n", ""))
assert OLD_SCHEMA != NEW_SCHEMA and "'import'" not in OLD_SCHEMA.split("CREATE TABLE annotations")[0]

JOB_IDS = (3, 7, 12)  # gaps on purpose: ids must be kept


def _make_old_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(OLD_SCHEMA)
    conn.execute("INSERT INTO projects VALUES (1, 'eot_crane', '2026-10-30T10:00:00Z')")
    conn.execute("INSERT INTO projects VALUES (2, 'other', '2026-10-30T10:00:00Z')")
    conn.execute("INSERT INTO images (id, project_id, file_name, path, width, height, uploaded_at,"
                 " review_status, roboflow_image_id, sha256) VALUES "
                 "(10, 1, 'a.jpg', 'p/a.jpg', 1280, 720, 't', 'done', 'rf10', 'aa'), "
                 "(11, 1, 'b.jpg', 'p/b.jpg', 1280, 720, 't', 'todo', NULL, 'bb')")
    conn.execute("INSERT INTO classes VALUES (5, 1, 'person', 'a worker', 0.05, 0.5)")
    jobs = [
        (3, 1, "detect", "done", 1.0, 2, 2, None, {"model": "fake", "image_ids": None,
                                                   "options": {}}, "t1", "t2", "t3"),
        (7, 1, "roboflow_push", "failed", 0.5, 1, 2, "RuntimeError: HTTP 401",
         {"statuses": ["accepted"]}, "t4", "t5", "t6"),
        (12, 2, "detect", "queued", 0.0, 0, 5, None, {"model": "pipeline"}, "t7", None, None),
    ]
    for j in jobs:
        conn.execute("INSERT INTO jobs (id, project_id, kind, status, progress, done, total, "
                     "error, config, created_at, started_at, finished_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (*j[:8], json.dumps(j[8]), *j[9:]))
    conn.execute("INSERT INTO annotations (id, image_id, class_id, x, y, w, h, mask, score, "
                 "source, status, job_id, reviewer, created_at, updated_at) VALUES "
                 "(100, 10, 5, 1, 2, 3, 4, NULL, 0.9, 'fake', 'auto_accepted', 3, NULL, 't', 't'), "
                 "(101, 10, 5, 5, 6, 7, 8, NULL, NULL, 'human', 'accepted', NULL, 'dhyan', 't', 't'), "
                 "(102, 11, 5, 9, 9, 9, 9, NULL, 0.3, 'fake', 'needs_review', 3, NULL, 't', 't')")
    conn.commit()
    conn.close()


def _dump(path) -> dict:
    conn = sqlite3.connect(path)
    try:
        out = {}
        for t in ("projects", "images", "classes", "jobs", "annotations"):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({t})")]
            out[t] = [dict(zip(cols, r)) for r in conn.execute(f"SELECT * FROM {t} ORDER BY id")]
        return out
    finally:
        conn.close()


def _jobs_sql(path) -> str:
    conn = sqlite3.connect(path)
    try:
        return conn.execute("SELECT sql FROM sqlite_master WHERE name='jobs'").fetchone()[0]
    finally:
        conn.close()


@pytest.fixture
def old_and_copy(tmp_path):
    original = tmp_path / "old" / "app.db"
    original.parent.mkdir()
    _make_old_db(original)
    copy = tmp_path / "copy" / "app.db"
    copy.parent.mkdir()
    shutil.copy2(original, copy)
    return original, copy


def test_rebuild_on_copy_keeps_every_row(old_and_copy):
    original, copy = old_and_copy
    before = _dump(original)
    db.init_db(copy)
    after = _dump(copy)

    assert _dump(original) == before  # the original was never touched
    assert "'import'" not in _jobs_sql(original)
    for t in ("images", "classes", "annotations"):
        assert after[t] == before[t], t
    # #30 added two project columns; old rows get NULL (= the defaults)
    assert after["projects"] == [{**p, "load_classes": None, "hook_radius": None}
                                 for p in before["projects"]]
    assert [j["id"] for j in after["jobs"]] == list(JOB_IDS)
    for old, new in zip(before["jobs"], after["jobs"]):
        assert new == {**old, "result": None}
    assert "'import'" in _jobs_sql(copy)
    assert "jobs_new" not in _jobs_sql(copy)


def test_rebuilt_db_works_and_second_run_is_noop(old_and_copy):
    _, copy = old_and_copy
    db.init_db(copy)
    first = _dump(copy)
    db.init_db(copy)
    assert _dump(copy) == first
    conn = db.connect(copy)
    assert db.migrate_jobs(conn) == "ok"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    job = db.create_job(conn, 1, "import", {"upload": "zip", "file_name": "x.zip"}, 3)
    assert job["kind"] == "import" and job["result"] is None and job["id"] == 13
    # annotations still point at jobs: FK enforcement on the new table
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO annotations (image_id, class_id, x, y, w, h, source, status, "
                     "job_id, created_at, updated_at) VALUES (10, 5, 1, 1, 1, 1, 'fake', "
                     "'needs_review', 999, 't', 't')")
    # ON DELETE CASCADE from projects still works on the rebuilt table
    with conn:
        conn.execute("DELETE FROM annotations")
        conn.execute("DELETE FROM projects WHERE id=2")
    assert conn.execute("SELECT COUNT(*) FROM jobs WHERE project_id=2").fetchone()[0] == 0


def test_failed_rebuild_rolls_back_and_keeps_old_table(old_and_copy, monkeypatch):
    _, copy = old_and_copy
    before = _dump(copy)
    calls = {"n": 0}

    def boom(conn):
        calls["n"] += 1
        if calls["n"] == 2:  # after the copy and the drop/rename
            raise RuntimeError("simulated failure")
        return set()
    monkeypatch.setattr(db, "_fk_problems", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        db.init_db(copy)
    assert _dump(copy) == before
    assert "'import'" not in _jobs_sql(copy)
    monkeypatch.undo()
    db.init_db(copy)  # a later start succeeds
    assert [j["id"] for j in _dump(copy)["jobs"]] == list(JOB_IDS)


def test_existing_fk_problems_do_not_block_rebuild(tmp_path):
    """An old database that already has a dangling reference is still migrated (not our row)."""
    path = tmp_path / "app.db"
    _make_old_db(path)
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO annotations (id, image_id, class_id, x, y, w, h, source, status, "
                 "job_id, created_at, updated_at) VALUES (200, 10, 77, 1, 1, 1, 1, 'fake', "
                 "'needs_review', NULL, 't', 't')")  # class 77 does not exist (FKs were off)
    conn.commit()
    conn.close()
    db.init_db(path)
    assert "'import'" in _jobs_sql(path)
    assert len(_dump(path)["annotations"]) == 4


def test_check_allows_import_but_no_result_column_uses_alter(tmp_path):
    path = tmp_path / "app.db"
    conn = sqlite3.connect(path)
    conn.executescript(NEW_SCHEMA.replace(
        "  result       TEXT,                      "
        "-- JSON summary; set by import jobs (#22) and examples-mode detect jobs, NULL otherwise\n", ""))
    conn.execute("INSERT INTO projects VALUES (1, 'p', 't')")
    conn.execute("INSERT INTO jobs (id, project_id, kind, status, config, created_at) "
                 "VALUES (4, 1, 'detect', 'done', '{}', 't')")
    conn.commit()
    conn.close()
    conn = db.connect(path)
    conn.row_factory = sqlite3.Row
    assert db.migrate_jobs(conn) == "added_result"
    conn.commit()
    assert db.job_dict(conn.execute("SELECT * FROM jobs").fetchone())["result"] is None


def test_new_db_has_result_and_import_kind(tmp_path):
    db.init_db(tmp_path / "app.db")
    conn = db.connect(tmp_path / "app.db")
    assert db.migrate_jobs(conn) == "ok"
    conn.execute("INSERT INTO projects (name, created_at) VALUES ('p', 't')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO jobs (project_id, kind, status, config, created_at) "
                     "VALUES (1, 'nonsense', 'queued', '{}', 't')")
    db.create_job(conn, 1, "import", {"upload": "files", "count": 0}, 0)
    job = conn.execute("SELECT * FROM jobs").fetchone()
    assert (job["kind"], job["status"]) == ("import", "queued")  # import with 0 still runs
