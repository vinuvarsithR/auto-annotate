"""Tests for the "yolo" detector. The model is mocked, except one slow test.

Slow test: skipped unless AUTOLABEL_SLOW=1. It downloads the stock yolo11n.pt (~5 MB, COCO
classes) into a temp folder and runs it on CPU on ultralytics' bundled bus.jpg, or on
AUTOLABEL_SLOW_IMAGE=path if set.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import autolabel.models as models
from autolabel import pipeline
from autolabel.models import get_detector, list_detectors
from autolabel.models.base import ClassSpec, Prediction
from autolabel.models.yolo import DEFAULT_MIN_SCORE, ENV_WEIGHTS, YoloDetector

CLASSES = [
    ClassSpec("person", "a worker wearing a hard hat, seen from above"),
    ClassSpec("rb", ""),
    ClassSpec("sb", "", min_score=0.3),
    ClassSpec("biax", ""),
]


class FakeBoxes:
    def __init__(self, rows):          # rows: (x1, y1, x2, y2, conf, cls)
        a = np.array(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]


class FakeResult:
    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """Stands in for ultralytics.YOLO: .names and .predict(), records the calls."""

    def __init__(self, names, rows):
        self.names = names
        self.rows = rows
        self.calls = []

    def predict(self, image, **kw):
        self.calls.append(kw)
        keep = [r for r in self.rows if r[4] >= kw["conf"] and int(r[5]) in kw["classes"]]
        return [FakeResult(keep)]


# Model order deliberately differs from the project order: matching must be by name.
NAMES = {0: "biax", 1: "forklift", 2: "person", 3: "sb", 4: "rb"}


@pytest.fixture
def image():
    return Image.new("RGB", (640, 480), "gray")


@pytest.fixture(autouse=True)
def clean_cache():
    models._CACHE.pop("yolo", None)
    yield
    models._CACHE.pop("yolo", None)


def test_classes_mapped_by_name_not_index(image):
    m = FakeModel(NAMES, [(10, 20, 40, 60, 0.9, 2), (0, 0, 5, 5, 0.8, 0), (1, 1, 2, 2, 0.7, 4)])
    preds = YoloDetector(model=m).detect(image, "img1", CLASSES)
    assert preds == [
        Prediction("img1", "person", (10.0, 20.0, 30.0, 40.0), 0.9, source="yolo"),
        Prediction("img1", "biax", (0.0, 0.0, 5.0, 5.0), 0.8, source="yolo"),
        Prediction("img1", "rb", (1.0, 1.0, 1.0, 1.0), 0.7, source="yolo"),
    ]
    assert all(p.mask is None for p in preds)


def test_model_classes_not_in_project_ignored(image):
    m = FakeModel(NAMES, [(0, 0, 5, 5, 0.9, 1)])            # forklift
    det = YoloDetector(model=m)
    assert det.detect(image, "i", CLASSES) == []
    assert m.calls[0]["classes"] == [0, 2, 3, 4]            # forklift never asked for


def test_project_classes_unknown_to_model(image):
    m = FakeModel({0: "person"}, [(0, 0, 5, 5, 0.9, 0)])
    preds = YoloDetector(model=m).detect(image, "i", [ClassSpec("crane", ""), ClassSpec("person", "")])
    assert [p.category for p in preds] == ["person"]


def test_no_matching_classes_skips_model(image):
    m = FakeModel({0: "car"}, [])
    assert YoloDetector(model=m).detect(image, "i", CLASSES) == []
    assert YoloDetector(model=m).detect(image, "i", []) == []
    assert m.calls == []


def test_per_class_min_score(image):
    m = FakeModel(NAMES, [(0, 0, 5, 5, 0.2, 3), (0, 0, 6, 6, 0.4, 3), (0, 0, 7, 7, 0.1, 2)])
    preds = YoloDetector(model=m).detect(image, "i", CLASSES)
    assert [(p.category, p.score) for p in preds] == [("sb", 0.4), ("person", 0.1)]
    assert m.calls[0]["conf"] == DEFAULT_MIN_SCORE          # lowest floor is asked for


def test_detector_min_score_option_and_cpu(image):
    m = FakeModel(NAMES, [(0, 0, 5, 5, 0.1, 2)])
    det = YoloDetector(model=m, min_score=0.2)
    assert det.detect(image, "i", CLASSES[:1]) == []
    kw = m.calls[0]
    assert kw["conf"] == 0.2 and kw["device"] == "cpu" and kw["verbose"] is False
    assert "imgsz" not in kw


def test_imgsz_option_and_list_names(image):
    m = FakeModel(["person", "rb"], [(0, 0, 5, 5, 0.9, 1)])  # names may be a list
    preds = YoloDetector(model=m, imgsz=1280).detect(image, "i", CLASSES)
    assert [p.category for p in preds] == ["rb"]
    assert m.calls[0]["imgsz"] == 1280


def test_score_clamped(image):
    m = FakeModel(NAMES, [(0, 0, 5, 5, 1.3, 2)])
    assert YoloDetector(model=m).detect(image, "i", CLASSES)[0].score == 1.0


def test_weights_required(monkeypatch):
    monkeypatch.delenv(ENV_WEIGHTS, raising=False)
    with pytest.raises(ValueError, match=ENV_WEIGHTS):
        YoloDetector()


def test_missing_weights_path(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_WEIGHTS, str(tmp_path / "nope" / "best.pt"))
    with pytest.raises(ValueError, match="not found"):
        YoloDetector()


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("AUTOLABEL_DETECTORS", raising=False)
    assert "yolo" not in list_detectors()
    with pytest.raises(ValueError, match="AUTOLABEL_DETECTORS=yolo"):
        get_detector("yolo")


def test_enabled_by_env_and_built_once(monkeypatch):
    monkeypatch.setenv("AUTOLABEL_DETECTORS", "owlv2,yolo")
    assert list_detectors() == ["fake", "owlv2", "yolo"]
    m = FakeModel(NAMES, [])
    d1 = get_detector("yolo", model=m)
    assert d1.name == "yolo" and d1.model is m
    assert get_detector("yolo") is d1


def test_pipeline_run_with_yolo(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOLABEL_DETECTORS", "yolo")
    path = tmp_path / "a.png"
    Image.new("RGB", (100, 50)).save(path)
    m = FakeModel(NAMES, [(1, 1, 11, 11, 0.6, 3)])
    results = list(pipeline.run([("7", str(path))], CLASSES, detector="yolo",
                                options={"model": m}))
    assert results == [("7", [Prediction("7", "sb", (1.0, 1.0, 10.0, 10.0), 0.6, source="yolo")])]


def test_importing_yolo_module_does_not_load_torch():
    code = ("import sys, autolabel.models.yolo; "
            "sys.exit(1 if ('torch' in sys.modules or 'ultralytics' in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@pytest.mark.skipif(os.environ.get("AUTOLABEL_SLOW") != "1", reason="slow: set AUTOLABEL_SLOW=1")
def test_real_stock_model_cpu(monkeypatch, tmp_path):
    import ultralytics
    monkeypatch.chdir(tmp_path)                       # yolo11n.pt downloads here
    path = os.environ.get("AUTOLABEL_SLOW_IMAGE") or str(
        Path(ultralytics.__file__).parent / "assets" / "bus.jpg")
    img = Image.open(path).convert("RGB")
    classes = [ClassSpec("person", ""), ClassSpec("rb", "")]   # rb is not a COCO class
    det = YoloDetector(weights="yolo11n.pt", device="cpu")
    det.detect(img, "warmup", classes)
    n, t0 = 5, time.perf_counter()
    for _ in range(n):
        preds = det.detect(img, "x", classes)
    secs = (time.perf_counter() - t0) / n
    print(f"\nYOLO11n CPU: {secs:.3f} s/image, {len(preds)} boxes, size {img.size}")
    w, h = img.size
    for p in preds:
        assert p.category == "person" and p.source == "yolo"
        assert DEFAULT_MIN_SCORE <= p.score <= 1
        x, y, bw, bh = p.bbox
        assert 0 <= x and 0 <= y and bw > 0 and bh > 0 and x + bw <= w + 1e-3 and y + bh <= h + 1e-3
    if not os.environ.get("AUTOLABEL_SLOW_IMAGE"):
        assert preds, "bus.jpg has people in it"
