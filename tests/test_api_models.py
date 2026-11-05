"""Endpoint 16: GET /api/models returns list_detectors() and each one's modes."""

from fastapi.testclient import TestClient

from app.main import create_app
from tests.test_api_helpers import make_client, setup_project


def test_models_returns_list_detectors(tmp_path):
    client, _ = make_client(tmp_path, detectors=("pipeline", "fake"))
    with client as c:
        r = c.get("/api/models")
        assert r.status_code == 200
        assert r.json() == {"models": ["pipeline", "fake"],
                            "modes": {"pipeline": [], "fake": ["text", "examples"]}}


def test_modes_have_one_key_per_model_default_first(tmp_path):
    modes = {"owlv2": ["text", "examples"], "yoloe": ["examples"]}
    client, _ = make_client(tmp_path, detectors=("owlv2", "yolo", "yoloe"), modes=modes)
    with client as c:
        assert c.get("/api/models").json() == {
            "models": ["owlv2", "yolo", "yoloe"],
            "modes": {"owlv2": ["text", "examples"], "yolo": [], "yoloe": ["examples"]}}


def test_modes_come_from_autolabel_detector_modes(tmp_path, monkeypatch):
    monkeypatch.setattr("autolabel.models.detector_modes", lambda name: ["a", name], raising=False)
    app = create_app(tmp_path / "data", start_worker=False, list_detectors=lambda: ["x"],
                     static_dir=None)
    with TestClient(app) as c:
        assert c.get("/api/models").json()["modes"] == {"x": ["a", "x"]}


def test_models_code_without_modes_counts_as_no_modes(tmp_path, monkeypatch):
    monkeypatch.delattr("autolabel.models.detector_modes", raising=False)
    app = create_app(tmp_path / "data", start_worker=False, list_detectors=lambda: ["x"],
                     static_dir=None)
    with TestClient(app) as c:
        assert c.get("/api/models").json()["modes"] == {"x": []}


def test_models_are_what_jobs_accepts(tmp_path):
    client, _ = make_client(tmp_path, detectors=("fake",))
    with client as c:
        s = setup_project(c, n_images=1)
        listed = c.get("/api/models").json()["models"]
        assert listed == ["fake"]
        ok = c.post(f"/api/projects/{s['pid']}/jobs", json={"model": listed[0]})
        assert ok.status_code == 202
        bad = c.post(f"/api/projects/{s['pid']}/jobs", json={"model": "pipeline"})
        assert bad.status_code == 400
        assert bad.json()["error"]["code"] == "unknown_model"
