"""Tests for the "owlv2" detector. The model is mocked, except one slow test.

Slow test: skipped unless AUTOLABEL_SLOW=1. It downloads OWLv2 (~620 MB) the first time.
Optional AUTOLABEL_SLOW_IMAGE=path runs it on a real crane frame instead of a blank image.
"""

import os
import subprocess
import sys
import time

import pytest
from PIL import Image

import autolabel.models as models
from autolabel.models import get_detector, list_detectors
from autolabel.models.base import ClassSpec, Prediction
from autolabel.models.owlv2 import DEFAULT_MIN_SCORE, Owlv2TextDetector, prompt_for
from autolabel.owlv2 import Detection
from autolabel import pipeline

CLASSES = [
    ClassSpec("person", "a worker wearing a hard hat, seen from above"),
    ClassSpec("rb", ""),                       # empty description -> prompt is the name
    ClassSpec("sb", "a long thin yellow steel lifting beam", min_score=0.3),
]


class FakeBackend:
    """Stands in for autolabel.owlv2.Owlv2Detector; records what it was asked."""

    def __init__(self, dets):
        self.dets = dets
        self.calls = []

    def detect(self, image, prompts, threshold=0.05):
        self.calls.append((prompts, threshold))
        return [d for d in self.dets if d.score >= threshold]


@pytest.fixture
def image():
    return Image.new("RGB", (640, 480), "gray")


@pytest.fixture(autouse=True)
def clean_cache():
    models._CACHE.pop("owlv2", None)
    yield
    models._CACHE.pop("owlv2", None)


def test_prompts_and_lowest_threshold(image):
    be = FakeBackend([])
    Owlv2TextDetector(backend=be).detect(image, "img1", CLASSES)
    prompts, threshold = be.calls[0]
    assert prompts == ["a worker wearing a hard hat, seen from above", "rb",
                       "a long thin yellow steel lifting beam"]
    assert threshold == DEFAULT_MIN_SCORE


def test_prompt_for_strips_whitespace():
    assert prompt_for(ClassSpec("biax", "   ")) == "biax"
    assert prompt_for(ClassSpec("biax", " a crossbar ")) == "a crossbar"


def test_output_format(image):
    be = FakeBackend([Detection(0, 0.8, (10, 20, 30, 40)), Detection(1, 0.07, (1, 2, 3, 4))])
    preds = Owlv2TextDetector(backend=be).detect(image, "img1", CLASSES)
    assert preds == [
        Prediction("img1", "person", (10.0, 20.0, 30.0, 40.0), 0.8, source="owlv2"),
        Prediction("img1", "rb", (1.0, 2.0, 3.0, 4.0), 0.07, source="owlv2"),
    ]
    for p in preds:
        assert isinstance(p, Prediction) and p.mask is None
        assert p.category in {c.name for c in CLASSES}
        assert 0 <= p.score <= 1


def test_per_class_min_score(image):
    be = FakeBackend([Detection(2, 0.2, (0, 0, 5, 5)), Detection(2, 0.4, (5, 5, 5, 5))])
    preds = Owlv2TextDetector(backend=be).detect(image, "i", CLASSES)
    assert [p.score for p in preds] == [0.4]          # sb floor is 0.3


def test_detector_min_score_option(image):
    be = FakeBackend([Detection(0, 0.1, (0, 0, 5, 5))])
    det = Owlv2TextDetector(backend=be, min_score=0.2)
    assert det.detect(image, "i", CLASSES[:1]) == []
    assert be.calls[0][1] == 0.2


def test_bad_label_and_score_clamped(image):
    be = FakeBackend([Detection(9, 0.9, (0, 0, 1, 1)), Detection(0, 1.2, (0, 0, 1, 1))])
    preds = Owlv2TextDetector(backend=be).detect(image, "i", CLASSES)
    assert [(p.category, p.score) for p in preds] == [("person", 1.0)]


def test_no_classes(image):
    be = FakeBackend([Detection(0, 0.9, (0, 0, 1, 1))])
    assert Owlv2TextDetector(backend=be).detect(image, "i", []) == []
    assert be.calls == []


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("AUTOLABEL_DETECTORS", raising=False)
    assert list_detectors() == ["fake"]
    with pytest.raises(ValueError, match="AUTOLABEL_DETECTORS=owlv2"):
        get_detector("owlv2")


def test_enabled_by_env_and_built_once(monkeypatch):
    monkeypatch.setenv("AUTOLABEL_DETECTORS", "owlv2")
    assert list_detectors() == ["fake", "owlv2"]
    be = FakeBackend([])
    d1 = get_detector("owlv2", backend=be)
    assert d1.name == "owlv2" and d1.backend is be
    assert get_detector("owlv2") is d1


def test_pipeline_run_with_owlv2(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOLABEL_DETECTORS", "owlv2")
    path = tmp_path / "a.png"
    Image.new("RGB", (100, 50)).save(path)
    be = FakeBackend([Detection(0, 0.6, (1, 1, 10, 10))])
    results = list(pipeline.run([("7", str(path))], CLASSES, detector="owlv2",
                                options={"backend": be}))
    assert results == [("7", [Prediction("7", "person", (1.0, 1.0, 10.0, 10.0), 0.6, source="owlv2")])]


def test_importing_owlv2_module_does_not_load_torch():
    code = ("import sys, autolabel.models.owlv2; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


@pytest.mark.skipif(os.environ.get("AUTOLABEL_SLOW") != "1", reason="slow: set AUTOLABEL_SLOW=1")
def test_real_model_one_image():
    path = os.environ.get("AUTOLABEL_SLOW_IMAGE")
    img = Image.open(path).convert("RGB") if path else Image.new("RGB", (640, 480), "gray")
    classes = [ClassSpec("person", "a worker wearing a hard hat, seen from above"),
               ClassSpec("biax", "a short yellow crossbar hanging from crane cables")]
    det = Owlv2TextDetector(device="cpu")
    det.detect(img, "warmup", classes)
    t0 = time.perf_counter()
    preds = det.detect(img, "x", classes)
    secs = time.perf_counter() - t0
    print(f"\nOWLv2 CPU: {secs:.2f} s/image, {len(preds)} boxes, size {img.size}")
    w, h = img.size
    for p in preds:
        assert p.category in {"person", "biax"} and p.source == "owlv2"
        assert DEFAULT_MIN_SCORE <= p.score <= 1
        x, y, bw, bh = p.bbox
        assert 0 <= x and 0 <= y and bw > 0 and bh > 0 and x + bw <= w + 1e-3 and y + bh <= h + 1e-3
