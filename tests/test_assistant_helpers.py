"""
Shared helpers for the Claude connector tests (no tests in here).

* hermetic_env: the app never reads the real .env (settings.ENV_FILE -> a missing temp file)
  and every connector variable is set or removed explicitly. Secrets are fake values.
* make_app / open_client: an app with fake detectors, an optional fake Anthropic client, and a
  TestClient that looks like a loopback client on localhost (so /mcp/ works without a token).
* load_fixture_data: the api_samples.json data (project 1 "eot_crane", classes 1-4, images 10-12
  with real small JPEG files, boxes 101-108, jobs 5, 6, 8, 9, 10, 12, the hook point of image 10's
  video, trained model eot_crane_v1), as docs/fixtures/assistant/tools.json expects.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from app.assistant import settings as assistant_settings
from app.db import db
from app.main import create_app

ROOT = Path(__file__).parents[1]
SAMPLES = json.loads((ROOT / "docs" / "fixtures" / "api_samples.json").read_text(encoding="utf-8"))
TOOLS_FIXTURE = json.loads((ROOT / "docs" / "fixtures" / "assistant" / "tools.json")
                           .read_text(encoding="utf-8"))
CHAT_FIXTURE = json.loads((ROOT / "docs" / "fixtures" / "assistant" / "chat_fake.json")
                          .read_text(encoding="utf-8"))

FAKE_KEY = "sk-ant-test-fake-key"
CONNECTOR_VARS = ("ANTHROPIC_API_KEY", "AUTOLABEL_MCP_TOKEN", "AUTOLABEL_CLAUDE_MODEL",
                  "AUTOLABEL_CLAUDE_EFFORT", "AUTOLABEL_CLAUDE_MAX_TOOL_CALLS",
                  "AUTOLABEL_CLAUDE_MAX_TOKENS")
DEFAULT_MODES = {"fake": ["text", "examples"]}
JOB_SAMPLES = ("GET /api/jobs/5", "GET /api/jobs/6", "GET /api/jobs/8 (import done)",
               "GET /api/jobs/9 (import done, folder upload)", "GET /api/jobs/10 (detect, examples)",
               "GET /api/jobs/12 (detect, examples, owlv2 + sam2)")


def hermetic_env(monkeypatch, tmp_path, key: str = FAKE_KEY, token: str = "", **extra) -> None:
    monkeypatch.setattr(assistant_settings, "ENV_FILE", tmp_path / "no-such.env")
    for name in CONNECTOR_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", key)
    monkeypatch.setenv("AUTOLABEL_MCP_TOKEN", token)
    for name, value in extra.items():
        monkeypatch.setenv(name, value)


def make_app(tmp_path, anthropic_client=None, detectors=("fake",), modes=None):
    modes = DEFAULT_MODES if modes is None else modes
    return create_app(tmp_path / "data", start_worker=False, static_dir=None,
                      list_detectors=lambda: list(detectors),
                      detector_modes=lambda name: list(modes.get(name, [])),
                      anthropic_client=anthropic_client)


def open_client(app, client_host: str = "127.0.0.1", host: str = "localhost:8000") -> TestClient:
    return TestClient(app, base_url=f"http://{host}", client=(client_host, 50000))


def connect(app):
    return db.connect(app.state.db_path)


def jpeg(size=(1280, 720)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(buf, format="JPEG")
    return buf.getvalue()


def load_fixture_data(app) -> None:
    """Insert the api_samples.json data with the fixture ids. The app must be started."""
    data_dir = Path(app.state.data_dir)
    project = SAMPLES["GET /api/projects/1"]
    classes = SAMPLES["GET /api/projects/1/classes"]["classes"]
    images = SAMPLES["GET /api/projects/1/images"]["images"]
    boxes = SAMPLES["all_annotations"]
    weights = SAMPLES["GET /api/projects/1/weights"]["weights"]
    point = SAMPLES["GET /api/projects/1/hook"]["points"][0]
    folder = data_dir / "uploads" / "1"
    folder.mkdir(parents=True, exist_ok=True)
    conn = connect(app)
    try:
        with conn:
            conn.execute("INSERT INTO projects (id, name, created_at) VALUES (?, ?, ?)",
                         (project["id"], project["name"], project["created_at"]))
            for c in classes:
                conn.execute("INSERT INTO classes (id, project_id, name, description, min_score, "
                             "accept_score) VALUES (?, 1, ?, ?, ?, ?)",
                             (c["id"], c["name"], c["description"], c["min_score"],
                              c["accept_score"]))
            for im in images:
                blob = jpeg((im["width"], im["height"]))
                path = folder / f"{im['id']}_{im['file_name']}"
                path.write_bytes(blob)
                conn.execute(
                    "INSERT INTO images (id, project_id, file_name, path, width, height, "
                    "uploaded_at, review_status, sha256) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)",
                    (im["id"], im["file_name"], str(path), im["width"], im["height"],
                     im["uploaded_at"], im["review_status"], f"sha-{im['id']}"))
            for key in JOB_SAMPLES:
                j = SAMPLES[key]
                conn.execute(
                    "INSERT INTO jobs (id, project_id, kind, status, progress, done, total, error, "
                    "config, result, created_at, started_at, finished_at) "
                    "VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (j["id"], j["kind"], j["status"], j["progress"], j["done"], j["total"],
                     j["error"], json.dumps(j["config"]),
                     json.dumps(j["result"]) if j["result"] is not None else None,
                     j["created_at"], j["started_at"], j["finished_at"]))
            for a in boxes:
                x, y, w, h = a["bbox"]
                conn.execute(
                    "INSERT INTO annotations (id, image_id, class_id, x, y, w, h, mask, polygon, "
                    "score, source, status, job_id, reviewer, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (a["id"], a["image_id"], a["class_id"], x, y, w, h,
                     json.dumps(a["mask"]) if a["mask"] is not None else None,
                     json.dumps(a["polygon"]) if a["polygon"] is not None else None,
                     a["score"], a["source"], a["status"], a["job_id"], a["reviewer"],
                     a["created_at"], a["updated_at"]))
            for w_ in weights:
                conn.execute(
                    "INSERT INTO trained_models (id, project_id, name, version, task, class_names, "
                    "imgsz, file_name, path, size_bytes, uploaded_at) "
                    "VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (w_["id"], w_["name"], w_["version"], w_["task"],
                     json.dumps(w_["class_names"]), w_["imgsz"], w_["file_name"],
                     str(data_dir / "weights" / "1" / "eot_crane_v1.pt"), w_["size_bytes"],
                     w_["uploaded_at"]))
            conn.execute("INSERT INTO hook_points (project_id, sequence, x, y, image_id, "
                         "updated_at) VALUES (1, ?, ?, ?, ?, ?)",
                         (point["sequence"], point["x"], point["y"], point["image_id"],
                          point["updated_at"]))
    finally:
        conn.close()


def add_images(app, n: int, start_id: int = 1000) -> None:
    """n more image rows for project 1 (no files: only counted, never read)."""
    conn = connect(app)
    try:
        with conn:
            conn.executemany(
                "INSERT INTO images (id, project_id, file_name, path, width, height, uploaded_at, "
                "review_status, sha256) VALUES (?, 1, ?, ?, 1280, 720, 't', 'todo', ?)",
                [(start_id + i, f"extra_{start_id + i:06d}.jpg", f"missing_{start_id + i}.jpg",
                  f"x{start_id + i}")
                 for i in range(n)])
    finally:
        conn.close()


def rows(app, sql: str, args=()) -> list[dict]:
    conn = connect(app)
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    finally:
        conn.close()


def same_keys(real, sample, where="$"):
    """The two JSON values have the same dict keys everywhere (lists: every item vs sample[0])."""
    if isinstance(sample, dict) and isinstance(real, dict):
        assert set(real) == set(sample), f"{where}: {sorted(set(real) ^ set(sample))}"
        for k in sample:
            same_keys(real[k], sample[k], f"{where}.{k}")
    elif isinstance(sample, list) and isinstance(real, list):
        if sample:
            for i, item in enumerate(real):
                same_keys(item, sample[0], f"{where}[{i}]")
