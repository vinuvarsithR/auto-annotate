"""Examples mode outside yoloe: fake, owlv2, detector_modes, pipeline. CPU only, mocked model,
except one slow test: skipped unless AUTOLABEL_SLOW=1. AUTOLABEL_SLOW_IMAGE=path picks its frame."""

import math
import os

import pytest
import torch
from PIL import Image

import autolabel.models as models
from autolabel import pipeline
from autolabel.models import detector_modes, get_detector
from autolabel.models.base import ClassSpec, Example, Prediction
from autolabel.models.owlv2 import TOP_PER_CLASS, Owlv2ExampleRun, Owlv2TextDetector
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
    """scenes: what image_guided_logits returns, one (logits, boxes) per call, in order."""

    def __init__(self, vectors=None, scenes=()):
        self.vectors = list(vectors or [])
        self.scenes = list(scenes)
        self.boxes, self.queries = [], None
        self.calls = 0

    def query_embedding(self, image, bbox):
        self.boxes.append(bbox)
        return torch.tensor(self.vectors.pop(0), dtype=torch.float32), 0.9

    def image_guided_logits(self, image, queries):
        self.queries = queries
        self.calls += 1
        return self.scenes.pop(0)


def scene(rows, boxes=None):
    """rows: logits [patches][classes]. boxes: pixels x, y, w, h on a 100 px image; default one
    far-apart 8 px box per patch, so NMS leaves them all alone."""
    boxes = boxes or [(i % 10 * 10, i // 10 * 10, 8, 8) for i in range(len(rows))]
    rel = [((x + w / 2) / 100, (y + h / 2) / 100, w / 100, h / 100) for x, y, w, h in boxes]
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(rel, dtype=torch.float32)


def owl_run(classes, cutoffs, scenes, floors=None):
    """A run with chosen cutoffs, to test detect on its own. prepare_examples has its own tests."""
    floors = floors or [0.05] * len(classes)
    return Owlv2ExampleRun(FakeOwl(scenes=scenes), classes, [torch.zeros(2) for _ in classes],
                           cutoffs, floors, {c.name: 1 for c in classes})


IM100 = Image.new("RGB", (100, 100))
PERSON, BIAX = ClassSpec("person", "p"), ClassSpec("biax", "b")


def test_owlv2_prototype_is_the_mean_of_normalized_embeddings(img):
    backend = FakeOwl([[2.0, 0.0], [0.0, 5.0]], [scene([[1.0], [2.0]], [(1, 2, 30, 20), (5, 6, 7, 8)])])
    run = Owlv2TextDetector(backend=backend).prepare_examples(
        CLASSES[:1], {"person": [ex(img, "person"), ex(img, "person", (5, 6, 7, 8))]})
    assert backend.boxes == [[1, 2, 30, 20], [5, 6, 7, 8]]     # x, y, w, h
    assert run.used == {"person": 2} and run.unusable == []
    assert run.prototypes[0].tolist() == pytest.approx([0.70710677, 0.70710677])


def test_owlv2_runs_one_query_per_class_with_examples(img):
    backend = FakeOwl([[1.0, 0.0], [0.0, 1.0]], [
        scene([[3.0, 3.0]], [(1, 2, 30, 20)]),                       # cutoffs, one example image
        scene([[3.0, -9.0], [-9.0, 3.0], [-9.0, 2.0]])])
    run = Owlv2TextDetector(backend=backend).prepare_examples(
        CLASSES, {"person": [ex(img, "person")], "biax": [ex(img, "biax")]})
    assert [c.name for c in run.classes] == ["person", "biax"]      # sb has no examples
    preds = run.detect(Image.new("RGB", (100, 100)), "3")
    assert len(backend.queries) == 2
    assert [(p.category, p.score, p.source, p.polygon) for p in preds] == [
        ("person", pytest.approx(0.5), "owlv2_image", None),
        ("biax", pytest.approx(0.5), "owlv2_image", None)]           # biax's 0.27 box: under its 0.5 floor
    assert preds[0].image_id == "3"


def test_owlv2_score_is_half_at_class_cutoff():
    run = owl_run([PERSON], [7.0], [scene([[7.0], [9.0]])])
    assert [p.score for p in run.detect(IM100, "1")] == pytest.approx([1 / (1 + math.exp(-2)), 0.5])


def test_owlv2_cutoff_is_median_of_example_logits(img):
    # Each example box matches its own patch exactly. The 9.0 patch belongs to no example.
    boxes = [(0, 0, 10, 10), (20, 0, 10, 10), (40, 0, 10, 10), (60, 0, 10, 10), (80, 0, 10, 10)]
    rows = [[1.0], [100.0], [5.0], [6.0], [9.0]]
    items = [ex(img, "person", b) for b in boxes[:4]]
    odd = Owlv2TextDetector(backend=FakeOwl([[1.0, 0.0]] * 3, [scene(rows, boxes)]))
    assert odd.prepare_examples([PERSON], {"person": items[:3]}).cutoffs == [5.0]
    even = Owlv2TextDetector(backend=FakeOwl([[1.0, 0.0]] * 4, [scene(rows, boxes)]))
    assert even.prepare_examples([PERSON], {"person": items}).cutoffs == [5.5]  # mean of 5 and 6


def test_owlv2_cutoff_runs_the_model_once_per_example_image(img, tmp_path):
    other = tmp_path / "other.png"
    Image.new("RGB", (100, 50)).save(other)
    backend = FakeOwl([[1.0, 0.0]] * 3, [scene([[2.0]], [(1, 2, 30, 20)]), scene([[4.0]], [(1, 2, 30, 20)])])
    run = Owlv2TextDetector(backend=backend).prepare_examples(
        [PERSON], {"person": [ex(img, "person"), ex(img, "person"), ex(str(other), "person")]})
    assert backend.calls == 2 and run.cutoffs == [2.0]


def test_owlv2_single_example_cutoff(img):
    backend = FakeOwl([[1.0, 0.0]], [scene([[-1.0], [4.5]], [(50, 50, 9, 9), (1, 2, 30, 20)])])
    run = Owlv2TextDetector(backend=backend).prepare_examples([PERSON], {"person": [ex(img, "person")]})
    assert run.cutoffs == [4.5]


def test_owlv2_each_class_has_its_own_cutoff(img):
    boxes = [(0, 0, 20, 20), (50, 50, 20, 20)]
    scenes = [scene([[3.0, 0.0], [0.0, 8.0]], boxes),   # logits of both prototypes, cutoffs 3 and 8
              scene([[3.0, 3.0]], [(0, 0, 9, 9)])]      # one logit of 3: typical person, weak biax
    run = Owlv2TextDetector(backend=FakeOwl([[1.0, 0.0], [0.0, 1.0]], scenes)).prepare_examples(
        [PERSON, BIAX], {"person": [ex(img, "person", boxes[0])], "biax": [ex(img, "biax", boxes[1])]})
    assert run.cutoffs == [3.0, 8.0]
    assert {p.category: p.score for p in run.detect(IM100, "1")} == pytest.approx({"person": 0.5})
    # 3 vs a cutoff of 8 is 0.0067, under the 0.05 floor. Lower the floor to see the score.
    run.floors = [0.05, 0.001]
    run.backend.scenes = [scene([[3.0, 3.0]], [(0, 0, 9, 9)])]
    assert {p.category: p.score for p in run.detect(IM100, "1")} == pytest.approx(
        {"person": 0.5, "biax": 1 / (1 + math.exp(5))})


def test_owlv2_min_score_uses_calibrated_score():
    # Raw sigmoid of 9.9 is 0.99995, calibrated it is 0.475: under a 0.5 floor.
    run = owl_run([PERSON], [10.0], [scene([[9.9], [10.0]])], floors=[0.5])
    assert [p.score for p in run.detect(IM100, "1")] == pytest.approx([0.5])
    # The default floor 0.05 is logit 2.94 under the cutoff.
    run = owl_run([PERSON], [10.0], [scene([[7.0], [7.1]])])
    assert [p.score for p in run.detect(IM100, "1")] == pytest.approx([1 / (1 + math.exp(2.9))])


def test_owlv2_cap_is_per_class_and_keeps_best():
    rows = [[float(i), float(i)] for i in range(8)]
    preds = owl_run([PERSON, BIAX], [0.0, 0.0], [scene(rows)]).detect(IM100, "1")
    assert TOP_PER_CLASS == 5
    for name in ("person", "biax"):
        kept = [p.score for p in preds if p.category == name]
        assert kept == pytest.approx([1 / (1 + math.exp(-i)) for i in (7, 6, 5, 4, 3)])


def test_owlv2_cap_keeps_order_where_the_score_rounds_to_one():
    rows = [[100.0 + i] for i in range(8)]
    boxes = [(i * 10, 0, 8, 8) for i in range(8)]
    preds = owl_run([PERSON], [0.0], [scene(rows, boxes)]).detect(IM100, "1")
    assert all(p.score == 1.0 for p in preds)
    assert [p.bbox[0] for p in preds] == pytest.approx([70, 60, 50, 40, 30])


def test_owlv2_cap_runs_after_nms():
    # Six near-copies of the best box, then 5 others. A cap before NMS would leave one box.
    boxes = [(10, 10, 20, 20)] * 6 + [(50 + i * 10, 60, 8, 8) for i in range(5)]
    rows = [[9.0 - i * 0.01] for i in range(6)] + [[5.0 - i] for i in range(5)]
    preds = owl_run([PERSON], [0.0], [scene(rows, boxes)]).detect(IM100, "1")
    assert [p.bbox[0] for p in preds] == pytest.approx([10, 50, 60, 70, 80])


def test_owlv2_cap_counts_only_boxes_inside_the_image():
    # Wide image: the first patches sit in the padding below it and clip to nothing.
    im = Image.new("RGB", (100, 50))
    boxes = [(i * 10, 70, 8, 8) for i in range(5)] + [(i * 15, 0, 8, 8) for i in range(5)]
    rows = [[9.0 - i * 0.1] for i in range(10)]
    preds = owl_run([PERSON], [0.0], [scene(rows, boxes)]).detect(im, "1")
    assert len(preds) == 5 and all(p.bbox[1] == 0 for p in preds)


def test_owlv2_no_box_above_floor_returns_empty():
    run = owl_run([PERSON], [10.0], [scene([[1.0], [2.0]])])
    assert run.detect(IM100, "1") == []


def test_owlv2_run_with_no_classes_finds_nothing():
    run = Owlv2TextDetector(backend=FakeOwl()).prepare_examples(CLASSES, {})
    assert run.classes == [] and run.detect(Image.new("RGB", (8, 8)), "1") == []


def test_owlv2_text_detect_still_uses_its_own_source():
    backend = FakeOwl()
    backend.detect = lambda image, prompts, threshold: [Detection(0, 0.7, (1, 2, 3, 4))]
    preds = Owlv2TextDetector(backend=backend).detect(Image.new("RGB", (8, 8)), "1", CLASSES)
    assert [p.source for p in preds] == ["owlv2"]


def test_owlv2_text_scores_unchanged():
    # Text mode: the backend's sigmoid is the score, no cutoff, no cap.
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    backend = FakeOwl()
    backend.detect = lambda image, prompts, threshold: [Detection(0, s, (i, 0, 3, 4)) for i, s in enumerate(scores)]
    preds = Owlv2TextDetector(backend=backend).detect(Image.new("RGB", (8, 8)), "1", CLASSES[:1])
    assert [p.score for p in preds] == pytest.approx(scores)


@pytest.mark.skipif(os.environ.get("AUTOLABEL_SLOW") != "1", reason="slow: set AUTOLABEL_SLOW=1")
def test_owlv2_real_model_boxes_per_image_are_capped(tmp_path):
    path = os.environ.get("AUTOLABEL_SLOW_IMAGE")
    if path:
        frame = Image.open(path).convert("RGB")
    else:   # busy noise: without the cap this leaves far more than 5 boxes per class
        frame = Image.effect_noise((640, 480), 80).convert("RGB")
    example = tmp_path / "example.png"
    frame.save(example)
    w, h = frame.size
    classes = [ClassSpec("a", "a"), ClassSpec("b", "b", min_score=0.001)]
    examples = {"a": [Example("1", str(example), "a", (w * 0.1, h * 0.1, w * 0.2, h * 0.2)),
                      Example("1", str(example), "a", (w * 0.6, h * 0.1, w * 0.2, h * 0.2))],
                "b": [Example("1", str(example), "b", (w * 0.4, h * 0.6, w * 0.3, h * 0.2))]}
    det = Owlv2TextDetector(device="cpu", min_score=0.001)
    run = det.prepare_examples(classes, examples)
    preds = run.detect(frame, "x")
    print(f"\nOWLv2 examples: {len(preds)} boxes, cutoffs {run.cutoffs}")
    assert len(preds) <= TOP_PER_CLASS * len(classes)
    for name in ("a", "b"):
        assert sum(p.category == name for p in preds) <= TOP_PER_CLASS
    assert all(p.source == "owlv2_image" and 0 <= p.score <= 1 for p in preds)

    # The logit method agrees with the existing scoring: same logits, same best sigmoid.
    logits, boxes = det.backend.image_guided_logits(frame, run.prototypes)
    assert boxes.shape == (logits.shape[0], 4) and logits.shape[1] == 2
    guided = det.backend.detect_image_guided(frame, [[p] for p in run.prototypes], threshold=0.0)
    for c in (0, 1):
        best = max(d.score for d in guided if d.label == c)
        assert best == pytest.approx(float(torch.sigmoid(logits[:, c]).max()), abs=1e-5)


def test_get_detector_fake_has_prepare_examples():
    assert callable(get_detector("fake").prepare_examples)
