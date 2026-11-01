"""Tests for autolabel/export/roboflow_push.py (CONTRACT #21). The SDK is mocked: no network."""

import json
import logging
import sys
import types
from pathlib import Path

import pytest

from autolabel.export import PushError, RoboflowSettings, push_image, roboflow_settings
from autolabel.export import roboflow_push

KEY = "rf_SECRET_test_key_123"
CLASSES = ["person", "rb", "sb", "biax"]
ENV = {"ROBOFLOW_API_KEY": KEY, "ROBOFLOW_WORKSPACE": "ws", "ROBOFLOW_PROJECT": "proj"}


class FakeProject:
    """Stands in for roboflow's Project; records calls, optionally misbehaves."""

    def __init__(self, fail=None, noisy=False, result_id="rf-img-1"):
        self.calls, self.fail, self.noisy, self.result_id = [], fail, noisy, result_id

    def single_upload(self, image_path=None, annotation_path=None, batch_name=None, **kw):
        self.calls.append({"name": Path(image_path).name, "bytes": Path(image_path).read_bytes(),
                           "annotation": annotation_path, "batch_name": batch_name, **kw})
        if self.noisy:   # an SDK that prints / logs the key must not leak it
            print(f"uploading with api_key={KEY}")
            print(f"stderr api_key={KEY}", file=sys.stderr)
            logging.getLogger("roboflow").warning("key=%s", "***")
        if self.fail:
            raise self.fail
        return {"image": {"id": self.result_id}, "annotation": {"success": True}}


@pytest.fixture
def fake_sdk(monkeypatch):
    """Replace the roboflow module with a fake; returns the FakeProject it hands out."""
    proj = FakeProject()
    opened = []

    class FakeRF:
        def __init__(self, api_key):
            opened.append(api_key)
            print(f"loading Roboflow workspace... key={api_key}")

        def workspace(self, ws):
            return types.SimpleNamespace(project=lambda p: proj)

    monkeypatch.setitem(sys.modules, "roboflow", types.SimpleNamespace(Roboflow=FakeRF))
    monkeypatch.setattr(roboflow_push, "_projects", {})
    proj.opened = opened
    return proj


@pytest.fixture
def settings():
    return roboflow_settings(ENV)


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "11_biax__cam0_00290.jpg"     # on-disk name differs, like the app
    path.write_bytes(b"jpegbytes")
    return {"id": 11, "file_name": "biax__cam0_00290.jpg", "path": str(path),
            "width": 1280, "height": 720}


BOXES = [{"class_name": "sb", "bbox": [300.0, 330.0, 690.0, 58.0]},
         {"class_name": "person", "bbox": [150.0, 210.0, 40.0, 50.0]}]


def no_key_anywhere(capsys, caplog, *texts):
    out, err = capsys.readouterr()
    for t in (out, err, caplog.text, *texts):
        assert KEY not in t


# ---- settings ----

def test_settings_ok(settings):
    assert isinstance(settings, RoboflowSettings)
    assert (settings.workspace, settings.project, settings.api_key) == ("ws", "proj", KEY)
    assert KEY not in repr(settings) and KEY not in str(settings)


def test_settings_missing_names_only():
    missing = roboflow_settings({"ROBOFLOW_API_KEY": KEY, "ROBOFLOW_WORKSPACE": "  "})
    assert missing == ["ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT"]


def test_settings_reads_dotenv_file(tmp_path, monkeypatch):
    for n in ENV:
        monkeypatch.delenv(n, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(f"ROBOFLOW_API_KEY={KEY}\nROBOFLOW_WORKSPACE=ws\nROBOFLOW_PROJECT=p\n")
    s = roboflow_settings(dotenv_path=env_file)
    assert s.project == "p"
    for n in ENV:
        monkeypatch.delenv(n, raising=False)   # don't leak into other tests


# ---- push_image ----

def test_push_uploads_image_and_boxes(fake_sdk, settings, image, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    rf_id = push_image(settings, image, BOXES, CLASSES, "autolabel_2026-11-01")
    assert rf_id == "rf-img-1"
    (call,) = fake_sdk.calls
    assert call["name"] == "biax__cam0_00290.jpg"          # original name, not 11_...
    assert call["bytes"] == b"jpegbytes"
    assert call["batch_name"] == "autolabel_2026-11-01"
    coco = json.loads(call["annotation"]["rawText"])
    assert coco["images"][0]["file_name"] == "biax__cam0_00290.jpg"
    assert [c["name"] for c in coco["categories"]] == CLASSES
    assert [(a["category_id"], a["bbox"]) for a in coco["annotations"]] == [
        (3, [300.0, 330.0, 690.0, 58.0]), (1, [150.0, 210.0, 40.0, 50.0])]
    no_key_anywhere(capsys, caplog, json.dumps(call["annotation"]))


def test_push_no_boxes(fake_sdk, settings, image):
    push_image(settings, image, [], CLASSES, "b")
    coco = json.loads(fake_sdk.calls[0]["annotation"]["rawText"])
    assert len(coco["images"]) == 1 and coco["annotations"] == []


def test_project_opened_once(fake_sdk, settings, image):
    push_image(settings, image, [], CLASSES, "b")
    push_image(settings, image, [], CLASSES, "b")
    assert len(fake_sdk.opened) == 1 and len(fake_sdk.calls) == 2


def test_push_unknown_class_is_error(fake_sdk, settings, image):
    with pytest.raises(PushError):
        push_image(settings, image, [{"class_name": "dog", "bbox": [0, 0, 1, 1]}], CLASSES, "b")
    assert fake_sdk.calls == []


def test_push_missing_file_is_error(fake_sdk, settings, image):
    image["path"] = image["path"] + ".gone"
    with pytest.raises(PushError):
        push_image(settings, image, [], CLASSES, "b")


def test_push_error_is_one_line_and_key_free(fake_sdk, settings, image, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    fake_sdk.fail = RuntimeError(f"POST https://api.roboflow.com/x?api_key={KEY}\nserver said no")
    fake_sdk.noisy = True
    with pytest.raises(PushError) as ei:
        push_image(settings, image, BOXES, CLASSES, "b")
    msg = str(ei.value)
    assert "\n" not in msg and "server said no" in msg and "***" in msg
    assert ei.value.__cause__ is None and ei.value.__suppress_context__
    no_key_anywhere(capsys, caplog, msg, repr(ei.value))


def test_push_noisy_sdk_output_is_swallowed(fake_sdk, settings, image, capsys, caplog):
    caplog.set_level(logging.DEBUG)
    fake_sdk.noisy = True
    push_image(settings, image, BOXES, CLASSES, "b")
    no_key_anywhere(capsys, caplog)


def test_push_missing_id_is_error(fake_sdk, settings, image):
    fake_sdk.result_id = None
    with pytest.raises(PushError):
        push_image(settings, image, [], CLASSES, "b")


def test_open_project_error_is_key_free(monkeypatch, settings, image, capsys, caplog):
    class FakeRF:
        def __init__(self, api_key):
            raise ValueError(f"invalid api key {api_key}")

    monkeypatch.setitem(sys.modules, "roboflow", types.SimpleNamespace(Roboflow=FakeRF))
    monkeypatch.setattr(roboflow_push, "_projects", {})
    with pytest.raises(PushError) as ei:
        push_image(settings, image, [], CLASSES, "b")
    no_key_anywhere(capsys, caplog, str(ei.value))
