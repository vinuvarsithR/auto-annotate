"""Tests for app/api/export.py: #14 export zip and #21 push route, plus one push job through the
real worker + real push_image with the roboflow SDK faked (no network)."""

import io
import json
import logging
import sys
import types
import zipfile
from pathlib import Path

import pytest

from autolabel.export import roboflow_push
from tests.test_api_helpers import make_client, run_job, setup_project

KEY = "rf_SECRET_test_key_123"
ENV = {"ROBOFLOW_API_KEY": KEY, "ROBOFLOW_WORKSPACE": "ws", "ROBOFLOW_PROJECT": "proj"}


@pytest.fixture(autouse=True)
def no_real_dotenv(tmp_path, monkeypatch):
    """Never read the repo's real .env in these tests; start with no Roboflow env vars."""
    monkeypatch.setattr(roboflow_push, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(roboflow_push, "_projects", {})
    for n in ENV:
        monkeypatch.delenv(n, raising=False)


@pytest.fixture
def env(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def _unzip(content: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


def _mark(c, iid, status):
    assert c.patch(f"/api/images/{iid}", json={"review_status": status}).status_code == 200


def _reject_all(c, iid):
    for a in c.get(f"/api/images/{iid}/annotations").json()["annotations"]:
        c.patch(f"/api/annotations/{a['id']}", json={"status": "rejected"})


def _project_with_reviews(c):
    """3 images, each with person auto_accepted + rb needs_review; i0 done, i1 empty, i2 todo."""
    s = setup_project(c, n_images=3)
    run_job(c, s["pid"])
    i0, i1, _ = s["image_ids"]
    _mark(c, i0, "done")
    _reject_all(c, i1)
    _mark(c, i1, "empty")
    return s


# ------------------------------------------------------------------ #14

def test_export_coco_default(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _project_with_reviews(c)
        r = c.post(f"/api/projects/{s['pid']}/export")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/zip"
        assert 'filename="eot_crane_coco.zip"' in r.headers["content-disposition"]
        files = _unzip(r.content)
        assert set(files) == {"annotations.json"}
        coco = json.loads(files["annotations.json"])
        assert [im["file_name"] for im in coco["images"]] == ["img_0.jpg", "img_1.jpg"]
        assert [c_["name"] for c_ in coco["categories"]] == ["person", "rb", "sb", "biax"]
        # only the auto_accepted person box of image 0 (rb is needs_review, image 1 is empty)
        assert [(a["image_id"], a["category_id"]) for a in coco["annotations"]] == [
            (s["image_ids"][0], 1)]


def test_export_yolo_with_images_and_statuses(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _project_with_reviews(c)
        r = c.post(f"/api/projects/{s['pid']}/export",
                   json={"format": "yolo", "statuses": ["needs_review"], "include_images": True})
        assert r.status_code == 200
        assert 'filename="eot_crane_yolo.zip"' in r.headers["content-disposition"]
        files = _unzip(r.content)
        assert {"data.yaml", "classes.txt", "labels/img_0.txt", "labels/img_1.txt",
                "images/img_0.jpg", "images/img_1.jpg"} <= set(files)
        assert "labels/img_2.txt" not in files and "images/img_2.jpg" not in files
        assert [l.split()[0] for l in files["labels/img_0.txt"].decode().splitlines()] == ["1"]
        assert files["labels/img_1.txt"] == b""
        stored = tmp_path / "data" / "uploads" / str(s["pid"]) / f"{s['image_ids'][0]}_img_0.jpg"
        assert files["images/img_0.jpg"] == stored.read_bytes()


@pytest.mark.parametrize("body", [
    {"format": "voc"},
    {"statuses": ["rejected"]},
    {"statuses": ["accepted", "rejected"]},
    {"statuses": ["nope"]},
    {"statuses": []},
    {"include_images": "maybe"},
])
def test_export_bad_body(tmp_path, body):
    with make_client(tmp_path)[0] as c:
        s = _project_with_reviews(c)
        r = c.post(f"/api/projects/{s['pid']}/export", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"


def test_export_errors(tmp_path):
    with make_client(tmp_path)[0] as c:
        assert c.post("/api/projects/99/export").json()["error"]["code"] == "not_found"
        s = setup_project(c, n_images=2)   # all todo
        r = c.post(f"/api/projects/{s['pid']}/export")
        assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"


def test_export_missing_file_is_404(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _project_with_reviews(c)
        for f in (tmp_path / "data" / "uploads" / str(s["pid"])).glob(f"{s['image_ids'][0]}_*"):
            f.unlink()
        r = c.post(f"/api/projects/{s['pid']}/export", json={"include_images": True})
        assert r.status_code == 404
        assert c.post(f"/api/projects/{s['pid']}/export").status_code == 200   # no images: fine


# ------------------------------------------------------------------ #21 route

def test_push_route_creates_job(tmp_path, env):
    with make_client(tmp_path)[0] as c:
        s = _project_with_reviews(c)
        r = c.post(f"/api/projects/{s['pid']}/roboflow/push")
        assert r.status_code == 202
        job = r.json()
        assert job["kind"] == "roboflow_push" and job["status"] == "queued"
        assert job["total"] == 2
        assert job["config"] == {"statuses": ["auto_accepted", "accepted", "edited"]}
        assert KEY not in r.text
        r = c.post(f"/api/projects/{s['pid']}/roboflow/push", json={"statuses": ["needs_review"]})
        assert r.json()["config"] == {"statuses": ["needs_review"]}


@pytest.mark.parametrize("body", [{"statuses": ["rejected"]}, {"statuses": ["accepted", "rejected"]},
                                  {"statuses": ["nope"]}, {"statuses": []}, {"statuses": "x"}])
def test_push_route_bad_statuses(tmp_path, env, body):
    with make_client(tmp_path)[0] as c:
        s = _project_with_reviews(c)
        r = c.post(f"/api/projects/{s['pid']}/roboflow/push", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"


def test_push_route_error_order(tmp_path, monkeypatch):
    with make_client(tmp_path)[0] as c:
        assert c.post("/api/projects/99/roboflow/push").status_code == 404
        s = setup_project(c, n_images=1)   # todo only
        pid = s["pid"]
        # bad statuses beats missing settings
        assert c.post(f"/api/projects/{pid}/roboflow/push",
                      json={"statuses": ["rejected"]}).status_code == 400
        # settings missing: names only
        monkeypatch.setenv("ROBOFLOW_API_KEY", KEY)
        r = c.post(f"/api/projects/{pid}/roboflow/push")
        assert r.status_code == 409
        msg = r.json()["error"]["message"]
        assert "ROBOFLOW_WORKSPACE" in msg and "ROBOFLOW_PROJECT" in msg and KEY not in msg
        # settings ok, but no done/empty image
        monkeypatch.setenv("ROBOFLOW_WORKSPACE", "ws")
        monkeypatch.setenv("ROBOFLOW_PROJECT", "proj")
        r = c.post(f"/api/projects/{pid}/roboflow/push")
        assert r.status_code == 409 and "no done or empty" in r.json()["error"]["message"]


def test_push_route_all_pushed_finishes_at_once(tmp_path, env):
    with make_client(tmp_path)[0] as c:
        s = _project_with_reviews(c)
        from app.db import db
        conn = db.connect(c.app.state.db_path)
        with conn:
            conn.execute("UPDATE images SET roboflow_image_id='x' WHERE project_id=?", (s["pid"],))
        conn.close()
        job = c.post(f"/api/projects/{s['pid']}/roboflow/push").json()
        assert job["status"] == "done" and job["total"] == 0


# ------------------------------------------------------------------ worker + push_image

class FakeProject:
    def __init__(self, fail=None):
        self.calls, self.fail = [], fail

    def single_upload(self, image_path=None, annotation_path=None, batch_name=None, **kw):
        print(f"SDK chatter api_key={KEY}")
        self.calls.append({"name": Path(image_path).name, "batch_name": batch_name,
                           "coco": json.loads(annotation_path["rawText"])})
        if self.fail:
            raise self.fail
        return {"image": {"id": f"rf-{len(self.calls)}"}}


def _fake_sdk(monkeypatch, proj):
    class FakeRF:
        def __init__(self, api_key):
            assert api_key == KEY

        def workspace(self, ws):
            assert ws == "ws"
            return types.SimpleNamespace(project=lambda p: proj)

    monkeypatch.setitem(sys.modules, "roboflow", types.SimpleNamespace(Roboflow=FakeRF))


def _rf_ids(c, pid):
    from app.db import db
    conn = db.connect(c.app.state.db_path)
    try:
        return [r[0] for r in conn.execute(
            "SELECT roboflow_image_id FROM images WHERE project_id=? ORDER BY id", (pid,))]
    finally:
        conn.close()


def test_push_job_through_real_worker(tmp_path, env, monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    proj = FakeProject()
    _fake_sdk(monkeypatch, proj)
    with make_client(tmp_path)[0] as c:       # worker uses the DEFAULT pusher = real push_image
        s = _project_with_reviews(c)
        job = c.post(f"/api/projects/{s['pid']}/roboflow/push").json()
        assert c.app.state.worker.run_once()
        job = c.get(f"/api/jobs/{job['id']}").json()
        assert job["status"] == "done" and job["done"] == 2 and job["error"] is None
        assert _rf_ids(c, s["pid"]) == ["rf-1", "rf-2", None]

        assert [call["name"] for call in proj.calls] == ["img_0.jpg", "img_1.jpg"]
        assert proj.calls[0]["batch_name"].startswith("autolabel_20")
        coco0, coco1 = proj.calls[0]["coco"], proj.calls[1]["coco"]
        assert [a["category_id"] for a in coco0["annotations"]] == [1]   # person only
        assert coco0["annotations"][0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
        assert coco1["annotations"] == []                                 # empty image

        # second push: nothing left, job done at once, SDK not called again
        job2 = c.post(f"/api/projects/{s['pid']}/roboflow/push").json()
        assert job2["status"] == "done" and job2["total"] == 0 and len(proj.calls) == 2
    out, err = capsys.readouterr()
    assert KEY not in out + err + caplog.text


def test_push_job_failure_through_real_worker_is_key_free(tmp_path, env, monkeypatch, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    proj = FakeProject(fail=RuntimeError(f"401 for https://api.roboflow.com/u?api_key={KEY}"))
    _fake_sdk(monkeypatch, proj)
    with make_client(tmp_path)[0] as c:
        s = _project_with_reviews(c)
        job = c.post(f"/api/projects/{s['pid']}/roboflow/push").json()
        assert c.app.state.worker.run_once()
        r = c.get(f"/api/jobs/{job['id']}")
        job = r.json()
        assert job["status"] == "failed" and "401" in job["error"]
        assert KEY not in r.text and "\n" not in job["error"]
        assert _rf_ids(c, s["pid"]) == [None, None, None]
    out, err = capsys.readouterr()
    assert KEY not in out + err + caplog.text
