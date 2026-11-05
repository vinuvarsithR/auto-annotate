"""Examples mode outside yoloe: fake, owlv2, detector_modes, pipeline. CPU only, no real model."""

import pytest
import torch
from PIL import Image

import autolabel.models as models
from autolabel import pipeline
from autolabel.models import detector_modes, get_detector
from autolabel.models.base import ClassSpec, Example, Prediction
from autolabel.models.owlv2 import Owlv2TextDetector
from autolabel.owlv2 import Detection

CLASSES = [ClassSpec("person", "p"), ClassSpec("biax", "b", min_score=0.5), ClassSpec("sb", "s")]


@pytest.fixture
def img(tmp_path):
    path = tmp_path / "im.png"
    Image.new("RGB", (100, 50)).save(path)
    return str(path)


def ex(path, category, bbox=(1, 2, 30, 20)):
    return Example("1", path, category, bbox)


def test_prediction_polygon_defaults_and_is_left_out_of_equality():
    a = Prediction("1", "person", (0, 0, 1, 1), 0.5, polygon=[[0, 0], [1, 0], [1, 1]])
    b = Prediction("1", "person", (0, 0, 1, 1), 0.5)
    assert b.polygon is None
    assert a == b and hash(a) == hash(b)


@pytest.mark.parametrize("name,modes", [("fake", ["text", "examples"]), ("owlv2", ["text", "examples"]),
                                        ("yoloe", ["examples"]), ("yolo", []), ("nope", [])])
def test_detector_modes(name, modes):
    assert detector_modes(name) == modes


def test_detector_modes_returns_a_copy():
    detector_modes("fake").append("x")
    assert detector_modes("fake") == ["text", "examples"]


# ---- fake ----

def test_fake_runs_only_classes_with_examples(img):
    run = pipeline.prepare_examples("fake", CLASSES, {"sb": [ex(img, "sb"), ex(img, "sb")]})
    assert [c.name for c in run.classes] == ["sb"]
    assert run.used == {"sb": 2} and run.unusable == []
    preds = run.detect(Image.new("RGB", (100, 100)), "9")
    assert [(p.category, p.source, p.score, p.image_id) for p in preds] == [("sb", "fake", 0.9, "9")]


def test_fake_unreadable_example_fails(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("nope")
    with pytest.raises(Exception):
        pipeline.prepare_examples("fake", CLASSES, {"sb": [ex(str(bad), "sb")]})


# ---- pipeline ----

def test_run_uses_the_example_run_and_skips_get_detector(img, monkeypatch):
    run = pipeline.prepare_examples("fake", CLASSES, {"person": [ex(img, "person")]})
    monkeypatch.setattr(pipeline, "get_detector", lambda *a, **k: pytest.fail("get_detector called"))
    results = list(pipeline.run([("5", img)], [], detector="not-a-detector", example_run=run))
    assert [(i, [p.category for p in preds]) for i, preds in results] == [("5", ["person"])]


def test_run_with_example_run_reports_progress(img):
    run = pipeline.prepare_examples("fake", CLASSES, {"person": [ex(img, "person")]})
    calls = []
    list(pipeline.run([("1", img), ("2", img)], [], example_run=run, on_progress=lambda d, t: calls.append((d, t))))
    assert calls == [(1, 2), (2, 2)]


def test_prepare_unknown_detector():
    with pytest.raises(ValueError, match="not available"):
        pipeline.prepare_examples("nope", CLASSES, {})


def test_prepare_detector_without_examples_mode(monkeypatch):
    monkeypatch.setenv("AUTOLABEL_DETECTORS", "yolo")
    with pytest.raises(ValueError, match="no examples mode"):
        pipeline.prepare_examples("yolo", CLASSES, {})


def test_prepare_passes_options_to_get_detector(img, monkeypatch):
    seen = {}
    real = models.get_detector
    monkeypatch.setattr(pipeline, "get_detector", lambda n, **o: seen.update(o) or real(n))
    pipeline.prepare_examples("fake", CLASSES, {}, options={"k": 1})
    assert seen == {"k": 1}


# ---- owlv2 ----

class FakeOwl:
    def __init__(self, vectors=None, dets=()):
        self.vectors = list(vectors or [])
        self.dets = list(dets)
        self.boxes, self.queries, self.threshold = [], None, None

    def query_embedding(self, image, bbox):
        self.boxes.append(bbox)
        return torch.tensor(self.vectors.pop(0), dtype=torch.float32), 0.9

    def detect_image_guided(self, image, queries, threshold=0.05):
        self.queries, self.threshold = queries, threshold
        return self.dets


def test_owlv2_prototype_is_the_mean_of_normalized_embeddings(img):
    backend = FakeOwl([[2.0, 0.0], [0.0, 5.0]])
    run = Owlv2TextDetector(backend=backend).prepare_examples(
        CLASSES[:1], {"person": [ex(img, "person"), ex(img, "person", (5, 6, 7, 8))]})
    assert backend.boxes == [[1, 2, 30, 20], [5, 6, 7, 8]]     # x, y, w, h
    assert run.used == {"person": 2} and run.unusable == []
    assert run.prototypes[0].tolist() == pytest.approx([0.70710677, 0.70710677])


def test_owlv2_runs_one_query_per_class_with_examples(img):
    dets = [Detection(0, 0.7, (1, 2, 3, 4)), Detection(1, 0.4, (5, 6, 7, 8)), Detection(1, 0.6, (9, 9, 9, 9))]
    backend = FakeOwl([[1.0, 0.0], [0.0, 1.0]], dets)
    run = Owlv2TextDetector(backend=backend).prepare_examples(
        CLASSES, {"person": [ex(img, "person")], "biax": [ex(img, "biax")]})
    assert [c.name for c in run.classes] == ["person", "biax"]      # sb has no examples
    preds = run.detect(Image.new("RGB", (64, 64)), "3")
    assert [len(q) for q in backend.queries] == [1, 1]
    assert backend.threshold == 0.05                                 # lowest floor
    assert [(p.category, p.score, p.source, p.polygon) for p in preds] == [
        ("person", pytest.approx(0.7), "owlv2_image", None),
        ("biax", pytest.approx(0.6), "owlv2_image", None)]           # biax's 0.5 floor applied
    assert preds[0].image_id == "3"


def test_owlv2_run_with_no_classes_finds_nothing():
    run = Owlv2TextDetector(backend=FakeOwl()).prepare_examples(CLASSES, {})
    assert run.classes == [] and run.detect(Image.new("RGB", (8, 8)), "1") == []


def test_owlv2_text_detect_still_uses_its_own_source():
    backend = FakeOwl()
    backend.detect = lambda image, prompts, threshold: [Detection(0, 0.7, (1, 2, 3, 4))]
    preds = Owlv2TextDetector(backend=backend).detect(Image.new("RGB", (8, 8)), "1", CLASSES)
    assert [p.source for p in preds] == ["owlv2"]


def test_get_detector_fake_has_prepare_examples():
    assert callable(get_detector("fake").prepare_examples)
