"""#9 `rerun` flag: re-run the last examples-mode job on the images nobody has reviewed yet.
Written by the integrator from docs/CONTRACT.md (#9, "Re-run with new examples"). Uses the fake
detector and the monkeypatched pipeline from test_worker_examples, so no GPU or weights."""

import json

import pytest

from app.db import db
from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_api_helpers import setup_project
from tests.test_worker_examples import _insert, _job, _rows, client, models  # noqa: F401 (fixtures)

RERUN_SAMPLE = "POST /api/projects/1/jobs (rerun examples)"


def _rerun(c, pid, **extra):
    return c.post(f"/api/projects/{pid}/jobs", json={"rerun": True, **extra})


def _review(c, iid, value):
    r = c.patch(f"/api/images/{iid}", json={"review_status": value})
    assert r.status_code == 200, r.text


def _set_review_status_in_db(c, iid, value):
    conn = db.connect(c.app.state.db_path)
    try:
        with conn:
            conn.execute("UPDATE images SET review_status=? WHERE id=?", (value, iid))
    finally:
        conn.close()


def _insert_job(c, pid, config):
    conn = db.connect(c.app.state.db_path)
    try:
        with conn:
            return conn.execute(
                "INSERT INTO jobs (project_id, kind, status, progress, done, total, config, "
                "created_at) VALUES (?, 'detect', 'done', 1, 0, 0, ?, 't')",
                (pid, json.dumps(config))).lastrowid
    finally:
        conn.close()


def _with_examples(c):
    """Project of 3 images with one human `person` box on the first, so `person` runs."""
    s = setup_project(c, n_images=3)
    _insert(c, s["image_ids"][0], s["class_ids"]["person"])
    return s


def _model_boxes(c, iid):
    return [a for a in _rows(c, iid) if a["source"] not in ("human", "import")]


# ------------------------------------------------------------------ request and config

def test_rerun_copies_settings_and_picks_todo_images(client, models):
    s = _with_examples(client)
    pid, (i0, i1, i2) = s["pid"], s["image_ids"]
    first = _job(client, pid, options={"k": 1})
    assert client.app.state.worker.run_once()
    _review(client, i0, "done")

    r = _rerun(client, pid)
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["kind"] == "detect" and job["status"] == "queued" and job["result"] is None
    assert job["config"] == {"model": "fake", "mode": "examples", "image_ids": [i1, i2],
                             "options": {"k": 1}, "rerun_of": first["id"]}
    assert job["total"] == 2
    same_keys({**job, "config": {**job["config"], "options": {}}}, FIXTURES[RERUN_SAMPLE])
    assert client.get(f"/api/jobs/{job['id']}").json()["config"] == job["config"]


def test_empty_images_are_not_rerun(client, models):
    s = _with_examples(client)
    pid, (i0, i1, i2) = s["pid"], s["image_ids"]
    _job(client, pid)
    assert client.app.state.worker.run_once()
    for a in _rows(client, i1):
        client.patch(f"/api/annotations/{a['id']}", json={"status": "rejected"})
    _review(client, i1, "empty")
    assert _rerun(client, pid).json()["config"]["image_ids"] == [i0, i2]


def test_source_is_latest_examples_job_and_reruns_chain(client, models):
    s = _with_examples(client)
    pid = s["pid"]
    _job(client, pid, options={"v": 1})
    second = _job(client, pid, options={"v": 2})
    client.post(f"/api/projects/{pid}/jobs", json={"model": "fake", "mode": "text"})  # not examples
    while client.app.state.worker.run_once():
        pass

    a = _rerun(client, pid).json()
    assert a["config"]["rerun_of"] == second["id"] and a["config"]["options"] == {"v": 2}
    b = _rerun(client, pid).json()  # the queued re-run is now the latest examples job
    assert b["config"]["rerun_of"] == a["id"] and b["config"]["options"] == {"v": 2}


def test_failed_examples_job_can_be_the_source(client, models):
    s = _with_examples(client)
    models.prepare_error = RuntimeError("boom")
    failed = _job(client, s["pid"])
    assert client.app.state.worker.run_once()
    assert client.get(f"/api/jobs/{failed['id']}").json()["status"] == "failed"
    assert _rerun(client, s["pid"]).json()["config"]["rerun_of"] == failed["id"]


def test_rerun_false_is_a_normal_job(client, models):
    s = _with_examples(client)
    r = client.post(f"/api/projects/{s['pid']}/jobs",
                    json={"model": "fake", "mode": "examples", "rerun": False})
    assert r.status_code == 202, r.text
    assert r.json()["config"] == {"model": "fake", "mode": "examples", "image_ids": None,
                                  "options": {}}


@pytest.mark.parametrize("extra", [{"model": "fake"}, {"mode": "examples"}, {"mode": None},
                                   {"image_ids": None}, {"image_ids": [1]}, {"options": {}}])
def test_rerun_takes_no_other_fields(client, models, extra):
    s = _with_examples(client)
    _job(client, s["pid"])
    r = _rerun(client, s["pid"], **extra)
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"
    same_keys(r.json(), FIXTURES["error example (400 bad_request, #9 rerun with other fields)"])


# ------------------------------------------------------------------ errors

def test_no_examples_job_is_409(client, models):
    s = _with_examples(client)
    client.post(f"/api/projects/{s['pid']}/jobs", json={"model": "fake", "mode": "text"})
    r = _rerun(client, s["pid"])
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    same_keys(r.json(), FIXTURES["error example (409 conflict, #9 rerun with no examples-mode job)"])


def test_examples_job_of_another_project_does_not_count(client, models):
    a = _with_examples(client)
    _job(client, a["pid"])
    b = setup_project(client, n_images=1, name="other")
    assert _rerun(client, b["pid"]).status_code == 409


def test_no_todo_image_is_409(client, models):
    s = _with_examples(client)
    _job(client, s["pid"])
    assert client.app.state.worker.run_once()
    for iid in s["image_ids"]:
        _review(client, iid, "done")
    r = _rerun(client, s["pid"])
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    same_keys(r.json(), FIXTURES["error example (409 conflict, #9 rerun with no todo images)"])


def test_source_detector_no_longer_listed_is_unknown_model(client, models):
    s = _with_examples(client)
    _insert_job(client, s["pid"], {"model": "owlv2", "mode": "examples", "image_ids": None,
                                   "options": {}})
    r = _rerun(client, s["pid"])
    assert r.status_code == 400 and r.json()["error"]["code"] == "unknown_model"


def test_unknown_project_is_404(client, models):
    assert _rerun(client, 999).status_code == 404


# ------------------------------------------------------------------ worker behaviour

def test_rerun_replaces_unreviewed_model_boxes_on_todo_images_only(client, models):
    s = _with_examples(client)
    pid, (i0, i1, i2) = s["pid"], s["image_ids"]
    first = _job(client, pid)
    assert client.app.state.worker.run_once()
    old = {iid: _model_boxes(client, iid) for iid in (i0, i1, i2)}
    assert all(len(v) == 1 and v[0]["status"] == "needs_review" for v in old.values())

    _review(client, i0, "done")  # reviewed image: never touched
    client.patch(f"/api/annotations/{old[i1][0]['id']}", json={"status": "accepted"})  # reviewed box

    job = _rerun(client, pid).json()
    assert client.app.state.worker.run_once()
    assert client.get(f"/api/jobs/{job['id']}").json()["status"] == "done"

    # i0 (done): the first job's box is still there, nothing new
    assert [a["id"] for a in _model_boxes(client, i0)] == [old[i0][0]["id"]]
    assert [a["source"] for a in _rows(client, i0) if a["source"] == "human"] == ["human"]
    # i1: the accepted box stays; the new box overlaps it (IoU 1) and is dropped: no duplicate
    b1 = _model_boxes(client, i1)
    assert [(a["id"], a["status"]) for a in b1] == [(old[i1][0]["id"], "accepted")]
    # i2: the old unreviewed box is gone, exactly one new box from the re-run
    b2 = _model_boxes(client, i2)
    assert [a["job_id"] for a in b2] == [job["id"]]  # ids may be reused by SQLite
    assert old[i2][0]["job_id"] == first["id"]


def test_rerun_uses_current_examples_from_all_images(client, models):
    s = _with_examples(client)
    pid, (i0, i1, _i2) = s["pid"], s["image_ids"]
    _job(client, pid)
    assert client.app.state.worker.run_once()
    rb = s["class_ids"]["rb"]
    _insert(client, i0, rb, status="accepted", source="human", bbox=(100, 100, 50, 50))
    _insert(client, i1, rb, status="edited", source="owlv2", bbox=(300, 100, 50, 50))
    _review(client, i0, "done")  # an example on a reviewed image still teaches

    _rerun(client, pid)
    assert client.app.state.worker.run_once()
    sent = models.prepare_calls[-1]["examples"]
    assert len(sent["rb"]) == 2 and len(sent["person"]) == 1


def test_rerun_result_is_the_usual_examples_result(client, models):
    s = _with_examples(client)
    _job(client, s["pid"])
    assert client.app.state.worker.run_once()
    job = _rerun(client, s["pid"]).json()
    assert client.app.state.worker.run_once()
    got = client.get(f"/api/jobs/{job['id']}").json()
    assert got["result"]["mode"] == "examples"
    assert got["result"]["examples_by_class"] == {"person": 1}
    assert {x["class_name"] for x in got["result"]["skipped_classes"]} == {"rb", "sb", "biax"}
    assert set(got["result"]) == set(FIXTURES["GET /api/jobs/10 (detect, examples)"]["result"])


def test_image_reviewed_during_rerun_is_not_written(client, models):
    s = _with_examples(client)
    pid, (_i0, i1, i2) = s["pid"], s["image_ids"]
    _job(client, pid)
    assert client.app.state.worker.run_once()
    before = {iid: [a["job_id"] for a in _model_boxes(client, iid)] for iid in (i1, i2)}

    job = _rerun(client, pid).json()
    models.on_run = lambda: _set_review_status_in_db(client, i1, "done")
    assert client.app.state.worker.run_once()
    got = client.get(f"/api/jobs/{job['id']}").json()
    assert got["status"] == "done" and got["done"] == got["total"] == 3  # i1 still counts
    assert [a["job_id"] for a in _model_boxes(client, i1)] == before[i1]  # untouched
    assert [a["job_id"] for a in _model_boxes(client, i2)] == [job["id"]]  # replaced
