"""Tests for autolabel/export/roboflow_push.py (CONTRACT #21). The SDK is mocked: no network."""

import copy
import json
import logging
import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import pytest

from autolabel.export import PushError, batch_name, open_project, push_image, roboflow_settings

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[1] / "docs" / "fixtures" / "api_samples.json").read_text()
)
KEY = "rf_SECRET_test_key_123"


class FakeProject:
    """Stands in for roboflow's Project; records calls, optionally misbehaves."""

    def __init__(self, fail=None, noisy=False, result_id="rf-img-1"):
        self.calls = []
        self.fail = fail
        self.noisy = noisy
        self.result_id = result_id

    def single_upload(self, image_path=None, annotation_path=None, batch_name=None, **kw):
        self.calls.append(
            {"image_path": image_path, "name": Path(image_path).name,
             "bytes": Path(image_path).read_bytes(),
             "annotation": annotation_path, "batch_name": batch_name, **kw}
        )
        if self.noisy:   # an SDK that prints / logs the key must not leak it
            print(f"uploading with api_key={KEY}")
            print(f"stderr api_key={KEY}", file=sys.stderr)
        if self.fail:
            raise self.fail
        return {"image": {"id": self.result_id}, "annotation": {"success": True}}


@pytest.fixture
def data(tmp_path):
    images = copy.deepcopy(FIXTURES["GET /api/projects/1/images"]["images"])
    classes = copy.deepcopy(FIXTURES["GET /api/projects/1/classes"]["classes"])
    anns = copy.deepcopy(FIXTURES["all_annotations"])
    by_id = {im["id"]: im for im in images}
    by_id[11]["review_status"] = "done"
    by_id[12]["review_status"] = "empty"
    path = tmp_path / "11_biax.jpg"     # on-disk name differs from file_name, like the app
    path.write_bytes(b"jpegbytes")
    return by_id, classes, anns, path


def no_key_anywhere(capsys, caplog, *texts):
    out, err = capsys.readouterr()
    for t in (out, err, caplog.text, *texts):
        assert KEY not in t


# ---- settings ----

def test_settings_ok():
    env = {"ROBOFLOW_API_KEY": KEY, "ROBOFLOW_WORKSPACE": "ws", "ROBOFLOW_PROJECT": "proj"}
    s, missing = roboflow_settings(env)
    assert missing == [] and (s.workspace, s.project, s.api_key) == ("ws", "proj", KEY)
    assert KEY not in repr(s) and KEY not in str(s)


def test_settings_missing_names_only():
    s, missing = roboflow_settings({"ROBOFLOW_API_KEY": KEY, "ROBOFLOW_WORKSPACE": "  "})
    assert s is None
    assert missing == ["ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT"]
    assert KEY not in " ".join(missing)


def test_settings_reads_dotenv_file(tmp_path, monkeypatch):
    for n in ("ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT"):
        monkeypatch.delenv(n, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"ROBOFLOW_API_KEY={KEY}\nROBOFLOW_WORKSPACE=ws\nROBOFLOW_PROJECT=p\n")
    s, missing = roboflow_settings(dotenv_path=env_file)
    assert missing == [] and s.project == "p"
    for n in ("ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT"):
        monkeypatch.delenv(n, raising=False)   # don't leak into other tests


def test_batch_name_is_utc_date():
    assert batch_name(datetime(2026, 11, 1, 23, 30, tzinfo=timezone.utc)) == "autolabel_2026-11-01"


# ---- push_image ----

def test_push_uploads_image_and_boxes(data, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    by_id, classes, anns, path = data
    proj = FakeProject()
    rf_id = push_image(proj, by_id[11], path, classes, anns, "autolabel_2026-11-01", secrets=[KEY])
    assert rf_id == "rf-img-1"
    (call,) = proj.calls
    assert call["name"] == by_id[11]["file_name"]          # original name, not 11_biax.jpg
    assert call["bytes"] == b"jpegbytes"
    assert call["batch_name"] == "autolabel_2026-11-01"
    coco = json.loads(call["annotation"]["rawText"])
    assert coco["images"][0]["file_name"] == by_id[11]["file_name"]
    assert sorted(a["id"] for a in coco["annotations"]) == [104, 105, 106]
    assert [c["name"] for c in coco["categories"]] == ["person", "rb", "sb", "biax"]
    no_key_anywhere(capsys, caplog, json.dumps(call["annotation"]))


def test_push_respects_statuses_and_never_rejected(data):
    by_id, classes, anns, path = data
    for a in anns:
        if a["image_id"] == 11:
            a["status"] = "rejected"
    anns.append(dict(anns[3], id=200, status="needs_review"))   # image 11, needs_review
    proj = FakeProject()
    push_image(proj, by_id[11], path, classes, anns, "b", statuses=["needs_review"])
    coco = json.loads(proj.calls[0]["annotation"]["rawText"])
    assert [a["id"] for a in coco["annotations"]] == [200]
    with pytest.raises(ValueError):
        push_image(proj, by_id[11], path, classes, anns, "b", statuses=["rejected"])


def test_push_empty_image_has_no_boxes(data):
    by_id, classes, anns, path = data
    proj = FakeProject()
    push_image(proj, by_id[12], path, classes, anns, "b")
    coco = json.loads(proj.calls[0]["annotation"]["rawText"])
    assert len(coco["images"]) == 1 and coco["annotations"] == []


def test_push_skips_already_pushed(data):
    by_id, classes, anns, path = data
    by_id[11]["roboflow_image_id"] = "old-id"
    proj = FakeProject()
    assert push_image(proj, by_id[11], path, classes, anns, "b") == "old-id"
    assert proj.calls == []


def test_push_refuses_todo_image(data):
    by_id, classes, anns, path = data
    proj = FakeProject()
    with pytest.raises(PushError):
        push_image(proj, by_id[10], path, classes, anns, "b")
    assert proj.calls == []


def test_push_error_is_one_line_and_key_free(data, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    by_id, classes, anns, path = data
    boom = RuntimeError(f"POST https://api.roboflow.com/x?api_key={KEY}\nserver said no")
    proj = FakeProject(fail=boom, noisy=True)
    with pytest.raises(PushError) as ei:
        push_image(proj, by_id[11], path, classes, anns, "b", secrets=[KEY])
    msg = str(ei.value)
    assert "\n" not in msg and "server said no" in msg and "***" in msg
    assert ei.value.__cause__ is None and ei.value.__suppress_context__
    no_key_anywhere(capsys, caplog, msg, repr(ei.value))


def test_push_noisy_sdk_output_is_swallowed(data, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    by_id, classes, anns, path = data
    push_image(FakeProject(noisy=True), by_id[11], path, classes, anns, "b", secrets=[KEY])
    no_key_anywhere(capsys, caplog)


def test_push_missing_id_is_error(data):
    by_id, classes, anns, path = data
    with pytest.raises(PushError):
        push_image(FakeProject(result_id=None), by_id[11], path, classes, anns, "b")


# ---- open_project (fake roboflow module, no network) ----

def test_open_project_uses_sdk_and_hides_key(monkeypatch, capsys, caplog):
    seen = {}

    class FakeRF:
        def __init__(self, api_key):
            seen["key"] = api_key
            print(f"loading Roboflow workspace... key={api_key}")

        def workspace(self, ws):
            seen["ws"] = ws
            return types.SimpleNamespace(project=lambda p: ("project", ws, p))

    monkeypatch.setitem(sys.modules, "roboflow", types.SimpleNamespace(Roboflow=FakeRF))
    s, _ = roboflow_settings({"ROBOFLOW_API_KEY": KEY, "ROBOFLOW_WORKSPACE": "ws", "ROBOFLOW_PROJECT": "p"})
    assert open_project(s) == ("project", "ws", "p")
    assert seen == {"key": KEY, "ws": "ws"}
    no_key_anywhere(capsys, caplog)


def test_open_project_error_is_key_free(monkeypatch, capsys, caplog):
    class FakeRF:
        def __init__(self, api_key):
            raise ValueError(f"invalid api key {api_key}")

    monkeypatch.setitem(sys.modules, "roboflow", types.SimpleNamespace(Roboflow=FakeRF))
    s, _ = roboflow_settings({"ROBOFLOW_API_KEY": KEY, "ROBOFLOW_WORKSPACE": "ws", "ROBOFLOW_PROJECT": "p"})
    with pytest.raises(PushError) as ei:
        open_project(s)
    no_key_anywhere(capsys, caplog, str(ei.value))
