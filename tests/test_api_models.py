"""Endpoint 16: GET /api/models returns exactly list_detectors()."""

from tests.test_api_helpers import make_client, setup_project


def test_models_returns_list_detectors(tmp_path):
    client, _ = make_client(tmp_path, detectors=("pipeline", "fake"))
    with client as c:
        r = c.get("/api/models")
        assert r.status_code == 200
        assert r.json() == {"models": ["pipeline", "fake"]}


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
