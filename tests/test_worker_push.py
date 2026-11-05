"""Tests for B2 (worker runs roboflow_push jobs) and B3 (db.create_job). SDK is faked."""

import pytest

from app.db import db
from tests.test_api_helpers import make_client, run_job, setup_project

SECRET = "rf_SECRET_test_key_123"


class FakePusher:
    """Stands in for autolabel.export.push_image; records every call."""

    def __init__(self, fail_on: int | None = None, error: str | None = None):
        self.calls = []
        self.fail_on = fail_on  # 1-based call number that raises
        self.error = error or f"HTTP 401 for https://api.roboflow.com/x?api_key={SECRET}\nmore"

    def push(self, image, boxes, class_names, batch_name):
        self.calls.append({"image": image, "boxes": boxes, "class_names": class_names,
                           "batch_name": batch_name})
        if self.fail_on is not None and len(self.calls) == self.fail_on:
            raise RuntimeError(self.error)
        return f"rf{image['id']}"

    def loader(self):
        return self.push, [SECRET]


def _client(tmp_path, pusher):
    c, _ = make_client(tmp_path)
    c.app.state.worker.load_pusher = pusher.loader
    return c


def _push_job(c, pid, statuses=None, total=None):
    conn = db.connect(c.app.state.db_path)
    try:
        if total is None:
            total = conn.execute(
                "SELECT COUNT(*) FROM images WHERE project_id=? AND roboflow_image_id IS NULL "
                "AND review_status IN ('done', 'empty')", (pid,)).fetchone()[0]
        config = {"statuses": statuses or ["auto_accepted", "accepted", "edited"]}
        return db.create_job(conn, pid, "roboflow_push", config, total)
    finally:
        conn.close()


def _rf_ids(c, pid):
    conn = db.connect(c.app.state.db_path)
    try:
        return [r[0] for r in conn.execute(
            "SELECT roboflow_image_id FROM images WHERE project_id=? ORDER BY id", (pid,))]
    finally:
        conn.close()


def _mark(c, iid, status):
    assert c.patch(f"/api/images/{iid}", json={"review_status": status}).status_code == 200


def _reject_all(c, iid):
    for a in c.get(f"/api/images/{iid}/annotations").json()["annotations"]:
        c.patch(f"/api/annotations/{a['id']}", json={"status": "rejected"})


# ------------------------------------------------------------------ B2

def test_push_only_done_and_empty_images_with_right_boxes(tmp_path):
    pusher = FakePusher()
    with _client(tmp_path, pusher) as c:
        s = setup_project(c, n_images=3)
        run_job(c, s["pid"])  # each image: person auto_accepted, rb needs_review
        i0, i1, _todo = s["image_ids"]
        _mark(c, i0, "done")
        _reject_all(c, i1)
        _mark(c, i1, "empty")
        job = _push_job(c, s["pid"])
        assert job["kind"] == "roboflow_push" and job["status"] == "queued" and job["total"] == 2
        assert c.app.state.worker.run_once()
        job = c.get(f"/api/jobs/{job['id']}").json()
        assert (job["status"], job["done"], job["total"], job["progress"]) == ("done", 2, 2, 1.0)
        assert [x["image"]["id"] for x in pusher.calls] == [i0, i1]
        # default statuses: auto_accepted yes, needs_review no
        assert pusher.calls[0]["boxes"] == [{"class_name": "person",
                                             "bbox": [10.0, 20.0, 30.0, 40.0],
                                             "polygon": None}]
        assert pusher.calls[1]["boxes"] == []  # empty image -> no labels
        assert pusher.calls[0]["class_names"] == ["person", "rb", "sb", "biax"]
        assert pusher.calls[0]["batch_name"] == f"autolabel_{job['started_at'][:10]}"
        assert pusher.calls[0]["image"]["width"] == 1280
        assert _rf_ids(c, s["pid"]) == [f"rf{i0}", f"rf{i1}", None]


def test_push_passes_polygon_with_each_box(tmp_path):
    pusher = FakePusher()
    with _client(tmp_path, pusher) as c:
        s = setup_project(c, n_images=1)
        iid = s["image_ids"][0]
        triangle = [[0.5, 0.25], [0.75, 0.25], [0.625, 0.5]]
        cid = s["class_ids"]["biax"]
        assert c.post(f"/api/images/{iid}/annotations",
                      json={"class_id": cid, "polygon": triangle}).status_code == 201
        assert c.post(f"/api/images/{iid}/annotations",
                      json={"class_id": cid, "bbox": [1, 2, 3, 4]}).status_code == 201
        _mark(c, iid, "done")
        assert c.app.state.worker.run_once() is False  # nothing queued yet
        _push_job(c, s["pid"])
        assert c.app.state.worker.run_once()
        polygon_box, plain_box = pusher.calls[0]["boxes"]
        assert polygon_box["polygon"] == triangle
        assert polygon_box["bbox"] == pytest.approx([640.0, 180.0, 320.0, 180.0])
        assert plain_box == {"class_name": "biax", "bbox": [1.0, 2.0, 3.0, 4.0], "polygon": None}


def test_statuses_from_config_are_used(tmp_path):
    pusher = FakePusher()
    with _client(tmp_path, pusher) as c:
        s = setup_project(c, n_images=1)
        run_job(c, s["pid"])
        _mark(c, s["image_ids"][0], "done")
        _push_job(c, s["pid"], statuses=["needs_review"])
        c.app.state.worker.run_once()
        assert [b["class_name"] for b in pusher.calls[0]["boxes"]] == ["rb"]


def test_already_pushed_images_are_skipped(tmp_path):
    pusher = FakePusher()
    with _client(tmp_path, pusher) as c:
        s = setup_project(c, n_images=2)
        for iid in s["image_ids"]:
            _mark(c, iid, "done")
        _push_job(c, s["pid"])
        c.app.state.worker.run_once()
        assert len(pusher.calls) == 2
        job = _push_job(c, s["pid"])  # nothing left: total 0 -> done at once
        assert job["status"] == "done" and job["total"] == 0 and job["progress"] == 1.0
        assert not c.app.state.worker.run_once()
        assert len(pusher.calls) == 2


def test_failure_keeps_earlier_ids_and_hides_key_then_rerun_finishes(tmp_path):
    pusher = FakePusher(fail_on=2)
    with _client(tmp_path, pusher) as c:
        s = setup_project(c, n_images=3)
        for iid in s["image_ids"]:
            _mark(c, iid, "done")
        job = _push_job(c, s["pid"])
        c.app.state.worker.run_once()
        job = c.get(f"/api/jobs/{job['id']}").json()
        assert job["status"] == "failed" and job["done"] == 1
        assert SECRET not in job["error"] and "\n" not in job["error"]
        assert "api_key=***" in job["error"]
        assert _rf_ids(c, s["pid"]) == [f"rf{s['image_ids'][0]}", None, None]
        # start again: only the 2 missing images are sent
        pusher.fail_on = None
        job2 = _push_job(c, s["pid"])
        assert job2["total"] == 2
        c.app.state.worker.run_once()
        assert c.get(f"/api/jobs/{job2['id']}").json()["status"] == "done"
        assert None not in _rf_ids(c, s["pid"])


def test_key_removed_even_without_api_key_param(tmp_path):
    pusher = FakePusher(fail_on=1, error=f"bad key {SECRET}")
    with _client(tmp_path, pusher) as c:
        s = setup_project(c, n_images=1)
        _mark(c, s["image_ids"][0], "done")
        job = _push_job(c, s["pid"])
        c.app.state.worker.run_once()
        error = c.get(f"/api/jobs/{job['id']}").json()["error"]
        assert SECRET not in error and "***" in error


def test_missing_settings_fail_the_job(tmp_path):
    def loader():
        raise RuntimeError("Roboflow settings missing from .env: ROBOFLOW_PROJECT")
    with make_client(tmp_path)[0] as c:
        c.app.state.worker.load_pusher = loader
        s = setup_project(c, n_images=1)
        _mark(c, s["image_ids"][0], "done")
        job = _push_job(c, s["pid"])
        c.app.state.worker.run_once()
        job = c.get(f"/api/jobs/{job['id']}").json()
        assert job["status"] == "failed" and "ROBOFLOW_PROJECT" in job["error"]


def test_detect_and_push_jobs_run_in_queue_order(tmp_path):
    pusher = FakePusher()
    with _client(tmp_path, pusher) as c:
        s = setup_project(c, n_images=1)
        _mark(c, s["image_ids"][0], "done")
        push = _push_job(c, s["pid"])
        detect = c.post(f"/api/projects/{s['pid']}/jobs", json={"model": "fake"}).json()
        assert c.app.state.worker.run_once()
        assert c.get(f"/api/jobs/{push['id']}").json()["status"] == "done"
        assert c.get(f"/api/jobs/{detect['id']}").json()["status"] == "queued"
        assert c.app.state.worker.run_once()
        assert c.get(f"/api/jobs/{detect['id']}").json()["status"] == "done"


# ------------------------------------------------------------------ B3

def test_create_job_queues_and_wakes(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = setup_project(c, n_images=1)["pid"]
        woke = []
        conn = db.connect(c.app.state.db_path)
        job = db.create_job(conn, pid, "roboflow_push", {"statuses": ["accepted"]}, 3,
                            wake=lambda: woke.append(1))
        conn.close()
        assert woke == [1]
        assert job == c.get(f"/api/jobs/{job['id']}").json()
        assert (job["status"], job["total"], job["done"], job["progress"]) == ("queued", 3, 0, 0)
        assert job["config"] == {"statuses": ["accepted"]} and job["started_at"] is None


def test_create_job_total_zero_is_done_at_once(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = setup_project(c, n_images=1)["pid"]
        woke = []
        conn = db.connect(c.app.state.db_path)
        job = db.create_job(conn, pid, "roboflow_push", {"statuses": []}, 0,
                            wake=lambda: woke.append(1))
        conn.close()
        assert job["status"] == "done" and job["progress"] == 1.0 and job["finished_at"]
        assert woke == []


def test_create_job_rejects_unknown_kind(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = setup_project(c, n_images=1)["pid"]
        conn = db.connect(c.app.state.db_path)
        with pytest.raises(ValueError):
            db.create_job(conn, pid, "export", {}, 1)
        conn.close()
