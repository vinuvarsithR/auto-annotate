"""migrate() on a database made before polygons: the column is added, no row is lost."""

import shutil
import sqlite3

import pytest

from app.db import db
from tests.test_api_helpers import make_client
from tests.test_db_migrate_jobs import OLD_SCHEMA as OLD_JOBS_SCHEMA

POLYGON_LINE = ("  polygon     TEXT,                       "
                "-- JSON [[x, y], ...] normalized 0..1, or NULL (plain box)\n")
SCHEMA_NO_POLYGON = db.SCHEMA.read_text(encoding="utf-8").replace(POLYGON_LINE, "")
SCHEMA_NO_POLYGON_OLD_JOBS = OLD_JOBS_SCHEMA.replace(POLYGON_LINE, "")
assert "polygon" not in SCHEMA_NO_POLYGON and "polygon" not in SCHEMA_NO_POLYGON_OLD_JOBS

SCHEMAS = {"only_polygon_missing": SCHEMA_NO_POLYGON,
           "older_than_import_too": SCHEMA_NO_POLYGON_OLD_JOBS}
TABLES = ("projects", "images", "classes", "jobs", "annotations")
MASK = '{"size": [720, 1280], "counts": "abc"}'


def _make_old_db(path, schema):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(schema)
    conn.execute("INSERT INTO projects VALUES (1, 'eot_crane', '2026-10-30T10:00:00Z')")
    conn.execute("INSERT INTO images (id, project_id, file_name, path, width, height, uploaded_at,"
                 " review_status, roboflow_image_id, sha256) VALUES "
                 "(10, 1, 'a.jpg', 'p/a.jpg', 1280, 720, 't', 'done', 'rf10', 'aa'), "
                 "(11, 1, 'b.jpg', 'p/b.jpg', 1280, 720, 't', 'todo', NULL, 'bb')")
    conn.execute("INSERT INTO classes VALUES (5, 1, 'person', 'a worker', 0.05, 0.5)")
    conn.execute("INSERT INTO jobs (id, project_id, kind, status, progress, done, total, error, "
                 "config, created_at) VALUES (3, 1, 'detect', 'done', 1.0, 2, 2, NULL, '{}', 't')")
    conn.execute("INSERT INTO annotations (id, image_id, class_id, x, y, w, h, mask, score, "
                 "source, status, job_id, reviewer, created_at, updated_at) VALUES "
                 f"(100, 10, 5, 1, 2, 3, 4, '{MASK}', 0.9, 'fake', 'auto_accepted', 3, NULL, 't', 't'), "
                 "(101, 10, 5, 5, 6, 7, 8, NULL, NULL, 'human', 'accepted', NULL, 'dhyan', 't', 't'), "
                 "(102, 11, 5, 9, 9, 9, 9, NULL, 0.3, 'fake', 'rejected', 3, NULL, 't', 't')")
    conn.commit()
    conn.close()


def _dump(path) -> dict:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        return {t: [dict(r) for r in conn.execute(f"SELECT * FROM {t} ORDER BY id")]
                for t in TABLES}
    finally:
        conn.close()


def _columns(path, table) -> list[str]:
    conn = sqlite3.connect(path)
    try:
        return [r[1] for r in conn.execute(f"PRAGMA table_info({table})")]
    finally:
        conn.close()


@pytest.mark.parametrize("schema", SCHEMAS.values(), ids=SCHEMAS.keys())
def test_migrate_adds_polygon_and_keeps_every_row(tmp_path, schema):
    old = tmp_path / "old.db"
    _make_old_db(old, schema)
    copy = tmp_path / "copy.db"
    shutil.copy(old, copy)
    assert "polygon" not in _columns(copy, "annotations")

    db.init_db(copy)

    assert "polygon" in _columns(copy, "annotations")
    before, after = _dump(old), _dump(copy)
    assert len(after["annotations"]) == 3
    for row in after["annotations"]:
        assert row.pop("polygon") is None  # old rows stay plain boxes
    for t in ("projects", "images", "classes", "annotations"):
        assert after[t] == before[t], t
    assert [(j["id"], j["kind"], j["status"]) for j in after["jobs"]] == [(3, "detect", "done")]


def test_migrate_twice_changes_nothing(tmp_path):
    path = tmp_path / "old.db"
    _make_old_db(path, SCHEMA_NO_POLYGON)
    db.init_db(path)
    once = _dump(path)
    db.init_db(path)
    assert _dump(path) == once
    assert _columns(path, "annotations").count("polygon") == 1


def test_fresh_database_has_polygon_in_the_contract_position(tmp_path):
    path = tmp_path / "new.db"
    db.init_db(path)
    cols = _columns(path, "annotations")
    assert cols.index("polygon") == cols.index("mask") + 1


def test_app_works_on_a_migrated_database(tmp_path):
    _make_old_db(tmp_path / "data" / "app.db", SCHEMA_NO_POLYGON)
    with make_client(tmp_path)[0] as c:
        rows = c.get("/api/images/10/annotations").json()["annotations"]
        assert [(a["id"], a["polygon"]) for a in rows] == [(100, None), (101, None)]
        assert rows[0]["mask"] == {"size": [720, 1280], "counts": "abc"}
        polygon = [[0.5, 0.25], [0.75, 0.25], [0.625, 0.5]]
        a = c.patch("/api/annotations/101", json={"polygon": polygon}).json()
        assert a["polygon"] == polygon and a["bbox"] == pytest.approx([640, 180, 320, 180])
