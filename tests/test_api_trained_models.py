"""#24 - #26 uploaded weights, plus the Train on Colab changes to #9, #16 and #18.
Written by the integrator from docs/CONTRACT.md. inspect_weights / register_trained /
unregister_trained are injected through create_app, so no ultralytics or weights are needed.

The fake inspect_weights reads the uploaded bytes as JSON: {"task", "names", "imgsz"};
bytes starting with b"bad" raise ValueError like an unloadable checkpoint."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.db import db
from app.main import create_app
from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_api_helpers import FakePipeline, setup_project
from tests.test_assistant_helpers import metrics_file

NAMES = ["person", "rb", "sb", "biax"]


class Info:
    def __init__(self, task, class_names, imgsz):
        self.task, self.class_names, self.imgsz = task, class_names, imgsz


class Hooks:
    def __init__(self):
        self.inspected, self.registered, self.unregistered = [], [], []

    def inspect(self, path, model=None):
        self.inspected.append(path)
        raw = Path(path).read_bytes()
        if raw.startswith(b"bad"):
            raise ValueError("not a loadable ultralytics checkpoint")
        d = json.loads(raw)
        return Info(d["task"], d["names"], d.get("imgsz"))

    def register(self, name, path, task, model=None):
        self.registered.append((name, path, task))

    def unregister(self, name):
        self.unregistered.append(name)


@pytest.fixture
def hooks():
    return Hooks()


@pytest.fixture
def pipe():
    return FakePipeline()


@pytest.fixture
def client(tmp_path, hooks, pipe):
    app = create_app(tmp_path / "data", start_worker=False, load_pipeline=pipe.loader,
                     list_detectors=lambda: ["fake"],
                     detector_modes=lambda n: {"fake": ["text", "examples"]}.get(n, []),
                     static_dir=None, inspect_weights=hooks.inspect,
                     register_trained=hooks.register, unregister_trained=hooks.unregister)
    with TestClient(app) as c:
        yield c


def pt(task="segment", names=NAMES, imgsz=960) -> bytes:
    return json.dumps({"task": task, "names": names, "imgsz": imgsz}).encode()


def upload(c, pid, content=None, task="segment", file_name="best.pt"):
    content = pt(task) if content is None else content
    return c.post(f"/api/projects/{pid}/weights",
                  files={"file": (file_name, content, "application/octet-stream")},
                  data={"task": task})


def weights_dir(c, pid):
    return Path(c.app.state.data_dir) / "weights" / str(pid)


def err(r, status, code):
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code
    return r.json()


# ------------------------------------------------------------------ #24 upload

def test_upload_stores_and_returns_trained_model(client, hooks):
    s = setup_project(client, n_images=1)
    r = upload(client, s["pid"])
    assert r.status_code == 201, r.text
    m = r.json()
    same_keys(m, FIXTURES["POST /api/projects/1/weights"])
    assert (m["project_id"], m["name"], m["version"], m["task"]) == (s["pid"], "eot_crane_v1", 1,
                                                                     "segment")
    assert m["class_names"] == NAMES and m["imgsz"] == 960 and m["file_name"] == "best.pt"
    assert m["size_bytes"] == len(pt())
    stored = weights_dir(client, s["pid"]) / "eot_crane_v1.pt"
    assert stored.read_bytes() == pt()
    assert list(weights_dir(client, s["pid"]).iterdir()) == [stored]  # no temp file left


def test_model_order_may_differ_and_is_kept(client):
    s = setup_project(client, n_images=1)
    r = upload(client, s["pid"], pt(names=["biax", "sb", "rb", "person"]))
    assert r.status_code == 201, r.text
    assert r.json()["class_names"] == ["biax", "sb", "rb", "person"]


def test_versions_are_never_reused(client):
    pid = setup_project(client, n_images=1)["pid"]
    v1 = upload(client, pid).json()
    v2 = upload(client, pid).json()
    assert (v1["name"], v2["name"]) == ("eot_crane_v1", "eot_crane_v2")
    assert client.delete(f"/api/weights/{v2['id']}").status_code == 204
    v3 = upload(client, pid).json()
    assert (v3["version"], v3["name"]) == (3, "eot_crane_v3")


def test_name_uses_safe_project_name(client):
    pid = setup_project(client, n_images=1, name="my crane!")["pid"]
    assert upload(client, pid).json()["name"] == "my_crane_v1"


def test_name_taken_by_another_project(client):
    a = setup_project(client, n_images=1)["pid"]
    assert upload(client, a).json()["name"] == "eot_crane_v1"
    client.patch(f"/api/projects/{a}", json={"name": "renamed"})
    b = setup_project(client, n_images=1)["pid"]  # a new project called eot_crane again
    assert upload(client, b).json()["name"] == f"eot_crane_p{b}_v1"
    assert client.get(f"/api/projects/{a}/weights").json()["weights"][0]["name"] == "eot_crane_v1"


def test_class_mismatch_reports_both_sides(client):
    pid = setup_project(client, n_images=1)["pid"]
    model_names = ["person", "rb", "beam", "biax"]
    body = err(upload(client, pid, pt(names=model_names)), 400, "class_mismatch")
    assert body["error"]["details"] == {"model_classes": model_names, "project_classes": NAMES,
                                        "missing_in_model": ["sb"], "extra_in_model": ["beam"]}
    same_keys(body, FIXTURES["error example (400 class_mismatch, #24)"])
    assert not weights_dir(client, pid).exists() or not any(weights_dir(client, pid).iterdir())
    assert client.get(f"/api/projects/{pid}/weights").json() == {"weights": []}


def test_class_mismatch_is_case_sensitive(client):
    pid = setup_project(client, n_images=1)["pid"]
    body = err(upload(client, pid, pt(names=["Person", "rb", "sb", "biax"])), 400, "class_mismatch")
    assert body["error"]["details"]["missing_in_model"] == ["person"]
    assert body["error"]["details"]["extra_in_model"] == ["Person"]


def test_repeated_model_name_is_a_mismatch(client):
    pid = setup_project(client, n_images=1)["pid"]
    body = err(upload(client, pid, pt(names=NAMES + ["rb"])), 400, "class_mismatch")
    assert body["error"]["details"]["missing_in_model"] == []
    assert body["error"]["details"]["extra_in_model"] == ["rb"]


def test_task_mismatch(client):
    pid = setup_project(client, n_images=1)["pid"]
    r = client.post(f"/api/projects/{pid}/weights",
                    files={"file": ("best.pt", pt("detect"), "application/octet-stream")},
                    data={"task": "segment"})
    same_keys(err(r, 400, "task_mismatch"), FIXTURES["error example (400 task_mismatch, #24)"])


@pytest.mark.parametrize("content", [b"bad bytes", pt("pose"), pt("classify")])
def test_bad_weights(client, content):
    pid = setup_project(client, n_images=1)["pid"]
    r = client.post(f"/api/projects/{pid}/weights",
                    files={"file": ("best.pt", content, "application/octet-stream")},
                    data={"task": "segment"})
    same_keys(err(r, 400, "bad_weights"), FIXTURES["error example (400 bad_weights, #24)"])
    assert client.get(f"/api/projects/{pid}/weights").json() == {"weights": []}


def test_bad_requests(client):
    pid = setup_project(client, n_images=1)["pid"]
    url = f"/api/projects/{pid}/weights"
    ok = ("best.pt", pt(), "application/octet-stream")
    err(client.post(url, data={"task": "segment"}), 400, "bad_request")                 # no file
    err(client.post(url, files={"file": ok}), 400, "bad_request")                       # no task
    err(client.post(url, files={"file": ok}, data={"task": "pose"}), 400, "bad_request")
    err(upload(client, pid, file_name="best.onnx"), 400, "bad_request")
    err(upload(client, pid, content=b""), 400, "bad_request")
    two = [("file", ok), ("file", ("other.pt", pt(), "application/octet-stream"))]
    err(client.post(url, files=two, data={"task": "segment"}), 400, "bad_request")


def test_upper_case_extension_is_fine(client):
    pid = setup_project(client, n_images=1)["pid"]
    assert upload(client, pid, file_name="BEST.PT").status_code == 201


def test_unknown_project_and_no_classes(client):
    err(upload(client, 999), 404, "not_found")
    pid = client.post("/api/projects", json={"name": "empty"}).json()["id"]
    err(upload(client, pid), 400, "no_classes")


# ------------------------------------------------------------------ #25 list, #26 delete

def test_list_in_version_order_without_deleted(client):
    pid = setup_project(client, n_images=1)["pid"]
    ids = [upload(client, pid).json()["id"] for _ in range(3)]
    client.delete(f"/api/weights/{ids[1]}")
    got = client.get(f"/api/projects/{pid}/weights").json()
    same_keys(got, FIXTURES["GET /api/projects/1/weights"])
    assert [w["version"] for w in got["weights"]] == [1, 3]
    err(client.get("/api/projects/999/weights"), 404, "not_found")


def test_delete_removes_file_and_unregisters(client, hooks):
    pid = setup_project(client, n_images=1)["pid"]
    m = upload(client, pid).json()
    r = client.delete(f"/api/weights/{m['id']}")
    assert (r.status_code, r.content) == (204, b"")
    assert FIXTURES["DELETE /api/weights/1"] is None
    assert not (weights_dir(client, pid) / "eot_crane_v1.pt").exists()
    assert hooks.unregistered == ["eot_crane_v1"]
    err(client.delete(f"/api/weights/{m['id']}"), 404, "not_found")
    err(client.delete("/api/weights/999"), 404, "not_found")


def test_delete_while_job_queued_is_409(client):
    pid = setup_project(client, n_images=1)["pid"]
    m = upload(client, pid).json()
    assert client.post(f"/api/projects/{pid}/jobs", json={"model": m["name"]}).status_code == 202
    err(client.delete(f"/api/weights/{m['id']}"), 409, "conflict")
    assert (weights_dir(client, pid) / "eot_crane_v1.pt").exists()


# ------------------------------------------------------------------ #16 and #9

def test_models_with_project_id(client):
    a = setup_project(client, n_images=1)["pid"]
    b = setup_project(client, n_images=1, name="other")["pid"]
    upload(client, a)
    got = client.get("/api/models", params={"project_id": a}).json()
    same_keys(got, FIXTURES["GET /api/models?project_id=1"])
    trained = got.pop("trained")
    assert got == {"models": ["fake", "eot_crane_v1"],
                   "modes": {"fake": ["text", "examples"], "eot_crane_v1": []}}
    assert [{k: t[k] for k in ("name", "source", "task", "metrics")} for t in trained] == [
        {"name": "eot_crane_v1", "source": "colab", "task": "segment", "metrics": None}]
    assert client.get("/api/models", params={"project_id": b}).json()["trained"] == []
    assert client.get("/api/models", params={"project_id": b}).json()["models"] == ["fake"]
    assert "trained" not in client.get("/api/models").json()  # no project_id: unchanged
    assert client.get("/api/models").json()["models"] == ["fake"]
    err(client.get("/api/models", params={"project_id": 999}), 404, "not_found")


def test_job_with_trained_model(client):
    pid = setup_project(client, n_images=1)["pid"]
    upload(client, pid)
    r = client.post(f"/api/projects/{pid}/jobs", json={"model": "eot_crane_v1"})
    assert r.status_code == 202, r.text
    assert r.json()["config"]["mode"] is None
    err(client.post(f"/api/projects/{pid}/jobs", json={"model": "eot_crane_v1", "mode": "examples"}),
        400, "unknown_mode")


def test_job_with_other_projects_or_deleted_model(client):
    a = setup_project(client, n_images=1)["pid"]
    b = setup_project(client, n_images=1, name="other")["pid"]
    m = upload(client, a).json()
    err(client.post(f"/api/projects/{b}/jobs", json={"model": m["name"]}), 400, "unknown_model")
    client.delete(f"/api/weights/{m['id']}")
    err(client.post(f"/api/projects/{a}/jobs", json={"model": m["name"]}), 400, "unknown_model")
    assert "eot_crane_v1" not in client.get("/api/models", params={"project_id": a}).json()["models"]


def test_worker_registers_then_runs(client, hooks, pipe):
    s = setup_project(client, n_images=2)
    upload(client, s["pid"])
    job = client.post(f"/api/projects/{s['pid']}/jobs",
                      json={"model": "eot_crane_v1", "options": {"k": 1}}).json()
    assert client.app.state.worker.run_once()
    got = client.get(f"/api/jobs/{job['id']}").json()
    assert got["status"] == "done", got["error"]
    path = str(weights_dir(client, s["pid"]) / "eot_crane_v1.pt")
    assert [(n, str(Path(p)), t) for n, p, t in hooks.registered] == [
        ("eot_crane_v1", str(Path(path)), "segment")]
    (call,) = pipe.calls
    assert call["detector"] == "eot_crane_v1" and call["options"] == {"k": 1}
    assert client.get(f"/api/images/{s['image_ids'][0]}/annotations").json()["annotations"]


def test_worker_fails_job_when_model_deleted_meanwhile(client, hooks, pipe):
    s = setup_project(client, n_images=1)
    m = upload(client, s["pid"]).json()
    job = client.post(f"/api/projects/{s['pid']}/jobs", json={"model": m["name"]}).json()
    conn = db.connect(client.app.state.db_path)
    try:
        with conn:
            conn.execute("UPDATE trained_models SET deleted_at='t' WHERE id=?", (m["id"],))
    finally:
        conn.close()
    assert client.app.state.worker.run_once()
    got = client.get(f"/api/jobs/{job['id']}").json()
    assert got["status"] == "failed" and "eot_crane_v1 was deleted" in got["error"]
    assert hooks.registered == [] and pipe.calls == []


# ------------------------------------------------------------------ #18

def test_delete_project_removes_weights(client, hooks):
    pid = setup_project(client, n_images=1)["pid"]
    upload(client, pid)
    upload(client, pid)
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    assert not weights_dir(client, pid).exists()
    assert sorted(hooks.unregistered) == ["eot_crane_v1", "eot_crane_v2"]


# ------------------------------------------------------------------ 2026-11-14: metrics part, source


def upload_with_metrics(c, pid, metrics: bytes, task="segment", name="eot_crane_segment_metrics.json"):
    return c.post(f"/api/projects/{pid}/weights",
                  files={"file": ("eot_crane_segment_best.pt", pt(task), "application/octet-stream"),
                         "metrics": (name, metrics, "application/json")},
                  data={"task": task})


def test_upload_with_metrics_stores_them(client, monkeypatch):
    pid = setup_project(client, n_images=1)["pid"]
    r = upload_with_metrics(client, pid, json.dumps(metrics_file()).encode())
    assert r.status_code == 201, r.text
    m = r.json()
    same_keys(m, FIXTURES["POST /api/projects/1/weights (with metrics)"])
    want = FIXTURES["metrics file (eot_crane_segment_metrics.json)"]
    assert (m["source"], m["metrics"], m["split"]) == ("colab", want["metrics"], want["split"])
    listed = client.get(f"/api/projects/{pid}/weights").json()["weights"]
    assert listed[0]["metrics"] == want["metrics"] and listed[0]["split"] == want["split"]
    trained = client.get("/api/models", params={"project_id": pid}).json()["trained"]
    assert trained[0]["metrics"] == want["metrics"] and trained[0]["source"] == "colab"


def test_upload_without_metrics_has_nulls(client):
    pid = setup_project(client, n_images=1)["pid"]
    m = upload(client, pid).json()
    assert (m["source"], m["metrics"], m["split"]) == ("colab", None, None)


def test_bad_metrics_stores_nothing_and_loads_nothing(client, hooks, monkeypatch):
    pid = setup_project(client, n_images=1)["pid"]
    bad = dict(metrics_file(), format="nope")
    body = err(upload_with_metrics(client, pid, json.dumps(bad).encode()), 400, "bad_metrics")
    assert body == FIXTURES["error example (400 bad_metrics, #24)"]
    err(upload_with_metrics(client, pid, json.dumps(metrics_file("detect")).encode()),
        400, "bad_metrics")  # task detect in the file, segment in the form
    err(upload_with_metrics(client, pid, b""), 400, "bad_metrics")
    err(upload_with_metrics(client, pid, b"{" + b" " * (1024 * 1024 + 1) + b"}"), 400,
        "bad_metrics")
    assert hooks.inspected == []  # checked before the weights are loaded
    assert client.get(f"/api/projects/{pid}/weights").json()["weights"] == []
    assert not weights_dir(client, pid).exists() or list(weights_dir(client, pid).iterdir()) == []
    assert [p.name for p in (Path(client.app.state.data_dir) / "weights").glob("staging_*")] == []


def test_metrics_part_rules(client, monkeypatch):
    pid = setup_project(client, n_images=1)["pid"]
    good = json.dumps(metrics_file()).encode()
    err(upload_with_metrics(client, pid, good, name="metrics.txt"), 400, "bad_request")
    r = client.post(f"/api/projects/{pid}/weights",
                    files=[("file", ("best.pt", pt(), "application/octet-stream")),
                           ("metrics", ("a_metrics.json", good, "application/json")),
                           ("metrics", ("b_metrics.json", good, "application/json"))],
                    data={"task": "segment"})
    err(r, 400, "bad_request")


def test_detect_metrics_with_null_mask(client, monkeypatch):
    pid = setup_project(client, n_images=1)["pid"]
    r = upload_with_metrics(client, pid, json.dumps(metrics_file("detect")).encode(), task="detect")
    assert r.status_code == 201, r.text
    assert r.json()["metrics"]["mask"] is None


def test_store_weights_copies_the_source(client, tmp_path):
    from app.api.weights import store_weights

    pid = setup_project(client, n_images=1)["pid"]
    src = tmp_path / "keep_best.pt"
    src.write_bytes(pt())
    conn = db.connect(client.app.state.db_path)
    try:
        m = store_weights(conn, client.app.state.data_dir, pid, src, src.name, "segment", "colab",
                          None, client.app.state.inspect_weights)
    finally:
        conn.close()
    assert src.read_bytes() == pt()  # the original stays
    assert (weights_dir(client, pid) / f"{m['name']}.pt").read_bytes() == pt()
    assert (m["file_name"], m["source"]) == ("keep_best.pt", "colab")
