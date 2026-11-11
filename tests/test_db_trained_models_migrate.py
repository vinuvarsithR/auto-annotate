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
