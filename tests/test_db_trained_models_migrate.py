"""Backend: migrate() adds the trained_models table (#24) to a database made before it."""

from app.db import db


def test_migrate_creates_trained_models_on_old_db(tmp_path):
    path = tmp_path / "app.db"
    db.init_db(path)
    conn = db.connect(path)
    try:
        conn.execute("DROP TABLE trained_models")  # a database from before Train on Colab
        conn.commit()
        db.migrate(conn)
        db.migrate(conn)  # running it again changes nothing
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(trained_models)")}
    finally:
        conn.close()
    assert {"project_id", "name", "version", "task", "class_names", "imgsz", "file_name",
            "path", "size_bytes", "uploaded_at", "deleted_at"} <= cols


def test_migrate_adds_source_metrics_split_to_old_rows(tmp_path):
    path = tmp_path / "app.db"
    db.init_db(path)
    conn = db.connect(path)
    try:
        conn.execute("DROP TABLE trained_models")
        conn.execute("""CREATE TABLE trained_models (
  id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, name TEXT NOT NULL UNIQUE,
  version INTEGER NOT NULL, task TEXT NOT NULL, class_names TEXT NOT NULL, imgsz INTEGER,
  file_name TEXT NOT NULL, path TEXT NOT NULL, size_bytes INTEGER NOT NULL,
  uploaded_at TEXT NOT NULL, deleted_at TEXT, UNIQUE (project_id, version))""")
        conn.execute("INSERT INTO projects (id, name, created_at) VALUES (1, 'p', 't')")
        conn.execute("INSERT INTO trained_models (project_id, name, version, task, class_names, "
                     "file_name, path, size_bytes, uploaded_at) VALUES "
                     "(1, 'p_v1', 1, 'detect', '[]', 'best.pt', 'x.pt', 1, 't')")
        conn.commit()
        db.migrate(conn)
        db.migrate(conn)  # again: nothing changes
        conn.commit()
        row = conn.execute("SELECT * FROM trained_models").fetchone()
        d = db.trained_model_dict(row)
        assert (d["source"], d["metrics"], d["split"]) == ("colab", None, None)
        import sqlite3
        import pytest
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE trained_models SET source='cloud'")
    finally:
        conn.close()
