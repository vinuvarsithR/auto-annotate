"""Tests for endpoints 9-10: create a job and poll it."""

from tests.test_api_helpers import FakePipeline, make_client, setup_project


def test_create_job_queued(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=3)
        r = c.post(f"/api/projects/{s['pid']}/jobs",
                   json={"model": "pipeline", "image_ids": None, "options": {}})
        assert r.status_code == 202
        job = r.json()
        assert job["kind"] == "detect" and job["status"] == "queued"
        assert (job["done"], job["total"], job["progress"]) == (0, 3, 0.0)
        assert job["config"] == {"model": "pipeline", "image_ids": None, "options": {}}
        assert job["started_at"] is None and job["finished_at"] is None
        assert c.get(f"/api/jobs/{job['id']}").json() == job


def test_job_runs_to_done(tmp_path):
    client, fake = make_client(tmp_path)
    with client as c:
        s = setup_project(c, n_images=2)
        jid = c.post(f"/api/projects/{s['pid']}/jobs",
                     json={"model": "fake", "options": {"x": 1}}).json()["id"]
        assert c.app.state.worker.run_once()
        assert not c.app.state.worker.run_once()  # nothing left
        job = c.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "done"
        assert (job["done"], job["total"], job["progress"]) == (2, 2, 1.0)
        assert job["started_at"] and job["finished_at"]
        call = fake.calls[0]
        assert call["detector"] == "fake" and call["options"] == {"x": 1}
        assert [img_id for img_id, _ in call["images"]] == [str(i) for i in s["image_ids"]]
        assert [cs.name for cs in call["classes"]] == ["person", "rb", "sb", "biax"]
        assert call["classes"][0].min_score == 0.05


def test_job_subset_of_images(tmp_path):
    client, fake = make_client(tmp_path)
    with client as c:
        s = setup_project(c, n_images=3)
        r = c.post(f"/api/projects/{s['pid']}/jobs", json={"image_ids": [s["image_ids"][1]]})
        assert r.json()["total"] == 1
        c.app.state.worker.run_once()
        assert [i for i, _ in fake.calls[0]["images"]] == [str(s["image_ids"][1])]


def test_job_failure_sets_one_line_error(tmp_path):
    client, _ = make_client(tmp_path, FakePipeline(fail_after=1))
    with client as c:
        s = setup_project(c, n_images=3)
        jid = c.post(f"/api/projects/{s['pid']}/jobs", json={}).json()["id"]
        c.app.state.worker.run_once()
        job = c.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed"
        assert job["error"] == "RuntimeError: CUDA out of memory while loading owlv2"
        assert job["done"] == 1 and job["finished_at"]


def test_pipeline_import_error_fails_job(tmp_path):
    def broken_loader():
        raise ModuleNotFoundError("No module named 'autolabel.pipeline'")

    from app.main import create_app
    from fastapi.testclient import TestClient
    app = create_app(tmp_path / "data", start_worker=False, load_pipeline=broken_loader,
                     list_detectors=lambda: ["pipeline"], static_dir=None)
    with TestClient(app) as c:
        s = setup_project(c, n_images=1)
        jid = c.post(f"/api/projects/{s['pid']}/jobs", json={}).json()["id"]
        c.app.state.worker.run_once()
        job = c.get(f"/api/jobs/{jid}").json()
        assert job["status"] == "failed" and "autolabel.pipeline" in job["error"]


def test_job_validation_errors(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = c.post("/api/projects", json={"name": "empty"}).json()["id"]
        r = c.post(f"/api/projects/{pid}/jobs", json={})
        assert (r.status_code, r.json()["error"]["code"]) == (400, "no_classes")

        s = setup_project(c, n_images=1, name="p2")
        r = c.post(f"/api/projects/{s['pid']}/jobs", json={"model": "yolo99"})
        assert (r.status_code, r.json()["error"]["code"]) == (400, "unknown_model")
        r = c.post(f"/api/projects/{s['pid']}/jobs", json={"image_ids": [9999]})
        assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_image_ids")
        assert c.post("/api/projects/77/jobs", json={}).status_code == 404
        assert c.get("/api/jobs/77").status_code == 404


def test_no_detectors_installed(tmp_path):
    def missing():
        raise ModuleNotFoundError("No module named 'autolabel.models'")

    client, _ = make_client(tmp_path)
    client.app.state.list_detectors = missing
    with client as c:
        s = setup_project(c, n_images=1)
        r = c.post(f"/api/projects/{s['pid']}/jobs", json={})
        assert r.status_code == 400
        assert "not installed" in r.json()["error"]["message"]


def test_background_thread_runs_job(tmp_path):
    """The real thread (not run_once) picks the job up after wake()."""
    import time

    from app.main import create_app
    from fastapi.testclient import TestClient
    fake = FakePipeline()
    app = create_app(tmp_path / "data", start_worker=True, load_pipeline=fake.loader,
                     list_detectors=lambda: ["pipeline"], static_dir=None)
    with TestClient(app) as c:
        s = setup_project(c, n_images=2)
        jid = c.post(f"/api/projects/{s['pid']}/jobs", json={}).json()["id"]
        deadline = time.time() + 10
        while c.get(f"/api/jobs/{jid}").json()["status"] != "done":
            assert time.time() < deadline, "worker thread did not finish the job"
            time.sleep(0.05)
