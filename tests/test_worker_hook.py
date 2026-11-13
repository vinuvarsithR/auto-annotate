"""OWLv2 + SAM 2, worker side: hook settings + points read at job start, PostProcess passed to
pipeline.run only for owlv2 examples jobs, result.masks + result.hook (docs/fixtures/owl_sam
cases.json `hook_warnings`). pipeline.run / prepare_examples are monkeypatched (no models, no SAM)."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import autolabel.models
import autolabel.models.base
import autolabel.pipeline
from app.db import db
from app.main import create_app
from app.worker.worker import hook_warnings
from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_api_helpers import CRANE_CLASSES, image_bytes
from tests.test_worker_examples import Models, StandInExample

CASES = json.loads((Path(__file__).parents[1] / "docs" / "fixtures" / "owl_sam" / "cases.json")
                   .read_text(encoding="utf-8"))["hook_warnings"]
RB = "rb_load__cam0_2026-09-29_18-34-11"
BIAX = "biax__cam0_2026-09-29_18-48-25"


@pytest.fixture
def models(monkeypatch):
    m = Models()
    monkeypatch.setattr(autolabel.pipeline, "prepare_examples", m.prepare, raising=False)
    monkeypatch.setattr(autolabel.pipeline, "run", m.run)
    monkeypatch.setattr(autolabel.models, "detector_modes",
                        lambda name: {"fake": ["text", "examples"],
                                      "owlv2": ["text", "examples"]}.get(name, []), raising=False)
    if not hasattr(autolabel.models.base, "Example"):
        monkeypatch.setattr(autolabel.models.base, "Example", StandInExample, raising=False)
    return m


@pytest.fixture
def client(tmp_path, models):
    app = create_app(tmp_path / "data", start_worker=False,
                     list_detectors=lambda: ["fake", "owlv2"], static_dir=None)
    with TestClient(app) as c:
        yield c


def _project(c, names):
    pid = c.post("/api/projects", json={"name": "crane"}).json()["id"]
    files = [("files", (n, image_bytes(), "image/jpeg")) for n in names]
    imgs = c.post(f"/api/projects/{pid}/images", files=files).json()["images"]
    cls = c.put(f"/api/projects/{pid}/classes", json={"classes": CRANE_CLASSES}).json()["classes"]
    return pid, {i["file_name"]: i["id"] for i in imgs}, {k["name"]: k["id"] for k in cls}


def _sql(c, sql, args=()):
    conn = db.connect(c.app.state.db_path)
    try:
        with conn:
            conn.execute(sql, args)
    finally:
        conn.close()


def _example(c, iid, cid):
    _sql(c, "INSERT INTO annotations (image_id, class_id, x, y, w, h, source, status, "
            "created_at, updated_at) VALUES (?, ?, 10, 20, 30, 40, 'human', 'accepted', 't', 't')",
         (iid, cid))


def _point(c, pid, seq, xy):
    _sql(c, "INSERT INTO hook_points (project_id, sequence, x, y, image_id, updated_at) "
            "VALUES (?, ?, ?, ?, NULL, 't')", (pid, seq, *xy))


def _run(c, pid, **body):
    r = c.post(f"/api/projects/{pid}/jobs", json=body)
    assert r.status_code == 202, r.text
    assert c.app.state.worker.run_once()
    return c.get(f"/api/jobs/{r.json()['id']}").json()


def _setup_case(c, case):
    pid, ids, cids = _project(c, case["images"])
    first = next(iter(ids.values()))
    for name in case["classes_ran"]:
        _example(c, first, cids[name])
    for seq, xy in case["points"].items():
        _point(c, pid, seq, xy)
    c.patch(f"/api/projects/{pid}/hook",
            json={"load_classes": case["load_classes"], "radius": case["radius"]})
    return pid, ids


# ------------------------------------------------------------------ cases.json hook_warnings

@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_hook_warnings_cases(client, models, case):
    pid, _ = _setup_case(client, case)
    job = _run(client, pid, model="owlv2", mode="examples")
    assert job["status"] == "done", job["error"]
    assert job["result"]["hook"] == case["result_hook"]
    assert [k.name for k in models.run_calls[0]["classes"]] == case["classes_ran"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_hook_warnings_function(case):
    hook = {"load_classes": case["load_classes"], "points": case["result_hook"]["points"],
            "job_sequences": {}}
    from autolabel.sequences import sequence_of
    for name in case["images"]:
        s = sequence_of(name)
        hook["job_sequences"][s] = hook["job_sequences"].get(s, 0) + 1
    assert hook_warnings(hook, case["classes_ran"]) == case["result_hook"]["warnings"]


# ------------------------------------------------------------------ PostProcess

def test_post_process_built_from_settings_and_points(client, models):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg", f"{RB}_00050.jpg", f"{BIAX}_00290.jpg"])
    _example(client, ids[f"{RB}_00049.jpg"], cids["rb"])
    _point(client, pid, RB, (616.0, 347.0))
    client.patch(f"/api/projects/{pid}/hook", json={"load_classes": ["rb", "biax"], "radius": 99})
    job = _run(client, pid, model="owlv2", mode="examples", masks="sam2")
    post = models.run_calls[0]["kwargs"]["post"]
    assert tuple(post.load_classes) == ("rb", "biax") and post.radius == 99.0
    assert post.masks == "sam2"
    assert post.hooks == {str(ids[f"{RB}_00049.jpg"]): (616.0, 347.0),
                          str(ids[f"{RB}_00050.jpg"]): (616.0, 347.0)}
    assert job["result"]["masks"] == "sam2"
    assert job["result"]["hook"] == {
        "load_classes": ["rb", "biax"], "radius": 99.0, "points": {RB: [616.0, 347.0]},
        "warnings": [{"sequence": BIAX, "images": 1, "reason": "no hook point"}]}


def test_hooks_only_for_images_in_the_job(client, models):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg", f"{BIAX}_00290.jpg"])
    _example(client, ids[f"{RB}_00049.jpg"], cids["rb"])
    _point(client, pid, RB, (1.0, 2.0))
    job = _run(client, pid, model="owlv2", mode="examples",
               image_ids=[ids[f"{BIAX}_00290.jpg"]])
    assert models.run_calls[0]["kwargs"]["post"].hooks == {}
    assert job["result"]["hook"]["points"] == {}  # RB has no image in this job
    assert job["result"]["hook"]["warnings"] == [
        {"sequence": BIAX, "images": 1, "reason": "no hook point"}]


def test_masks_absent_is_none_in_post_and_result(client, models):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg"])
    _example(client, ids[f"{RB}_00049.jpg"], cids["person"])
    job = _run(client, pid, model="owlv2", mode="examples")
    assert models.run_calls[0]["kwargs"]["post"].masks == "none"
    assert job["result"]["masks"] == "none"
    assert "masks" not in job["config"]


def test_defaults_used_when_settings_are_null(client, models):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg"])
    _example(client, ids[f"{RB}_00049.jpg"], cids["rb"])
    job = _run(client, pid, model="owlv2", mode="examples")
    post = models.run_calls[0]["kwargs"]["post"]
    assert (tuple(post.load_classes), post.radius) == (("rb", "sb", "biax"), 130.0)
    assert (job["result"]["hook"]["load_classes"], job["result"]["hook"]["radius"]) == (
        ["rb", "sb", "biax"], 130.0)


def test_empty_load_classes_gives_no_warnings(client, models):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg"])
    _example(client, ids[f"{RB}_00049.jpg"], cids["rb"])
    client.patch(f"/api/projects/{pid}/hook", json={"load_classes": []})
    job = _run(client, pid, model="owlv2", mode="examples")
    assert job["result"]["hook"]["warnings"] == []
    assert tuple(models.run_calls[0]["kwargs"]["post"].load_classes) == ()


def test_settings_read_once_at_job_start(client, models):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg"])
    _example(client, ids[f"{RB}_00049.jpg"], cids["rb"])
    _point(client, pid, RB, (5.0, 6.0))
    # a change while the images run must not reach this job
    models.on_run = lambda: _sql(client, "UPDATE hook_points SET x=999")
    job = _run(client, pid, model="owlv2", mode="examples")
    assert models.run_calls[0]["kwargs"]["post"].hooks == {str(ids[f"{RB}_00049.jpg"]): (5.0, 6.0)}
    assert job["result"]["hook"]["points"] == {RB: [5.0, 6.0]}


def test_no_class_left_still_writes_masks_and_hook(client, models):
    pid, _, _ = _project(client, [f"{RB}_00049.jpg"])
    job = _run(client, pid, model="owlv2", mode="examples", masks="sam2")
    assert job["status"] == "done" and models.run_calls == []
    assert job["result"]["masks"] == "sam2"
    assert job["result"]["hook"]["warnings"] == []  # no load class ran


@pytest.mark.parametrize("body", [
    {"model": "owlv2", "mode": "text"},
    {"model": "fake", "mode": "examples"},
    {"model": "fake", "mode": "text"},
])
def test_post_only_for_owlv2_examples(client, models, body):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg"])
    _example(client, ids[f"{RB}_00049.jpg"], cids["rb"])
    job = _run(client, pid, **body)
    assert job["status"] == "done", job["error"]
    assert "post" not in models.run_calls[0]["kwargs"]
    result = job["result"]
    assert result is None or ("masks" not in result and "hook" not in result)


def test_owlv2_examples_job_matches_fixture_keys(client, models):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg", f"{BIAX}_00290.jpg"])
    for name in ("person", "rb", "biax"):  # examples_by_class keys as in the fixture
        _example(client, ids[f"{RB}_00049.jpg"], cids[name])
    _point(client, pid, RB, (616.0, 347.0))
    job = _run(client, pid, model="owlv2", mode="examples", masks="sam2")
    same_keys(job, FIXTURES["GET /api/jobs/12 (detect, examples, owlv2 + sam2)"])


def test_rerun_passes_the_copied_masks(client, models):
    pid, ids, cids = _project(client, [f"{RB}_00049.jpg"])
    _example(client, ids[f"{RB}_00049.jpg"], cids["rb"])
    _run(client, pid, model="owlv2", mode="examples", masks="sam2")
    job = _run(client, pid, rerun=True)
    assert job["config"]["masks"] == "sam2"
    assert models.run_calls[1]["kwargs"]["post"].masks == "sam2"
    assert job["result"]["masks"] == "sam2"
