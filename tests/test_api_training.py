"""#37 POST /projects/{pid}/training: colab package (fake cuda_available), local without CUDA ->
503 no_gpu pointing to colab, local with CUDA -> 501, bad bodies -> 400, 404 first.

Until export's #23 change (epochs / imgsz / base_weights) is merged, export_shims stands in for
it and records what the route passed; after the merge the real route runs and the notebook in
the zip is checked instead."""

import io
import json
import re
import sys
import zipfile

import pytest

from app.assistant import downloads
from app.main import default_cuda_available
from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_assistant_helpers import (
    SAMPLES, export_has, export_shims, hermetic_env, load_fixture_data, make_app, open_client,
    rows,
)


def _app(tmp_path, monkeypatch, cuda=False):
    hermetic_env(monkeypatch, tmp_path)
    return make_app(tmp_path, cuda_available=lambda: cuda)


@pytest.fixture
def colab(tmp_path, monkeypatch):
    """(client, recorded #23 bodies) with the fixture data."""
    app = _app(tmp_path, monkeypatch)
    calls = export_shims(monkeypatch)
    with open_client(app) as c:
        load_fixture_data(app)
        yield c, calls


def post(c, body, pid=1):
    return c.post(f"/api/projects/{pid}/training", json=body)


def err(r, status, code):
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code, r.text
    return r.json()["error"]["message"]


# ------------------------------------------------------------------ colab

def test_colab_defaults_match_the_fixture(colab):
    c, calls = colab
    r = post(c, SAMPLES["POST /api/projects/1/training (body)"])
    assert r.status_code == 201, r.text
    got = r.json()
    same_keys(got, FIXTURES["POST /api/projects/1/training"])
    assert {k: got[k] for k in ("target", "task", "epochs", "imgsz", "base_weights")} == {
        "target": "colab", "task": "segment", "epochs": 50, "imgsz": 960,
        "base_weights": "yolo11s-seg.pt"}
    pkg = got["package"]
    assert pkg["download_url"].startswith("/api/assistant/downloads/")
    assert pkg["file_name"] == "eot_crane_train_segment.zip"
    blob = c.get(pkg["download_url"]).content
    assert len(blob) == pkg["size_bytes"]
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert json.loads(zf.read("split.json")) == got["split"]
        notebook = zf.read("train.ipynb").decode("utf-8")
    assert got["split"]["rule"] == "by_sequence"
    assert any("eot_crane_segment_best.pt" in s and "eot_crane_segment_metrics.json" in s
               for s in got["next_steps"])
    if export_has("train_options"):
        assert re.search(r"imgsz\s*=\s*960", notebook) and re.search(r"epochs\s*=\s*50", notebook)
    else:
        assert calls == [{"task": "segment", "epochs": 50, "imgsz": 960,
                          "base_weights": "yolo11s-seg.pt"}]


def test_colab_values_reach_the_package(colab):
    c, calls = colab
    body = {"target": "colab", "task": "detect", "epochs": 7, "imgsz": 640,
            "base_weights": "yolo11n.pt"}
    r = post(c, body)
    assert r.status_code == 201, r.text
    assert {k: r.json()[k] for k in body} == body
    if export_has("train_options"):
        blob = c.get(r.json()["package"]["download_url"]).content
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            nb = zf.read("train.ipynb").decode("utf-8")
        assert "yolo11n.pt" in nb and re.search(r"imgsz\s*=\s*640", nb)
        assert re.search(r"epochs\s*=\s*7", nb)
    else:
        assert calls == [{k: body[k] for k in ("task", "epochs", "imgsz", "base_weights")}]


def test_detect_default_base_weights(colab):
    c, _ = colab
    r = post(c, {"target": "colab", "task": "detect"})
    assert r.status_code == 201, r.text
    assert r.json()["base_weights"] == "yolo11s.pt"


def test_colab_keeps_the_23_conflicts(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch)
    export_shims(monkeypatch)
    with open_client(app) as c:
        pid = c.post("/api/projects", json={"name": "empty"}).json()["id"]
        err(post(c, {"target": "colab"}, pid), 409, "conflict")
        assert list((tmp_path / "data").glob("assistant/downloads/*.zip")) == []


# ------------------------------------------------------------------ local

def test_local_without_cuda_is_503_no_gpu(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, cuda=False)
    with open_client(app) as c:
        load_fixture_data(app)
        r = post(c, {"target": "local"})
        assert r.status_code == 503
        assert r.json() == SAMPLES["error example (503 no_gpu, #37)"]
        assert "colab" in r.json()["error"]["message"]
        assert not list(downloads.folder(app.state.data_dir).glob("*.zip"))


def test_local_with_cuda_is_501(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, cuda=True)
    with open_client(app) as c:
        load_fixture_data(app)
        r = post(c, {"target": "local", "task": "detect"})
        assert r.status_code == 501
        assert r.json() == SAMPLES["error example (501 not_implemented, #37)"]


def test_local_is_checked_after_the_body(tmp_path, monkeypatch):
    app = _app(tmp_path, monkeypatch, cuda=False)
    with open_client(app) as c:
        load_fixture_data(app)
        err(post(c, {"target": "local", "epochs": 0}), 400, "bad_request")


# ------------------------------------------------------------------ errors

@pytest.mark.parametrize("body", [
    None, [], {}, {"target": "gpu"}, {"target": "colab", "extra": 1},
    {"target": "colab", "task": "classify"},
    {"target": "colab", "epochs": 0}, {"target": "colab", "epochs": 501},
    {"target": "colab", "epochs": True}, {"target": "colab", "epochs": 5.0},
    {"target": "colab", "imgsz": 319}, {"target": "colab", "imgsz": 2080},
    {"target": "colab", "imgsz": 330},
    {"target": "colab", "task": "detect", "base_weights": "yolo11s-seg.pt"},
    {"target": "colab", "task": "segment", "base_weights": "yolo11s.pt"},
    {"target": "colab", "base_weights": "yolov8s.pt"},
])
def test_bad_body_is_400(colab, body):
    c, calls = colab
    if body is None:
        r = c.post("/api/projects/1/training")
    else:
        r = post(c, body)
    err(r, 400, "bad_request")
    assert calls == []


def test_unknown_project_is_404_before_the_body(colab):
    c, _ = colab
    err(post(c, {"target": "nope"}, pid=99), 404, "not_found")


def test_no_assistant_row_for_the_api(colab):
    c, _ = colab
    post(c, {"target": "colab"})
    assert rows(c.app, "SELECT * FROM assistant_actions") == []


# ------------------------------------------------------------------ default cuda_available

def test_default_cuda_available_without_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
    assert default_cuda_available() is False


def test_default_cuda_available_asks_torch(monkeypatch):
    class Cuda:
        @staticmethod
        def is_available():
            return True
    monkeypatch.setitem(sys.modules, "torch", type("T", (), {"cuda": Cuda})())
    assert default_cuda_available() is True
