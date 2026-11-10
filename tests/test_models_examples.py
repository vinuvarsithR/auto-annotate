"""Examples mode outside yoloe: fake, owlv2, detector_modes, pipeline. CPU only, mocked model,
except one slow test: skipped unless AUTOLABEL_SLOW=1. It needs crane frames
(AUTOLABEL_CRANE_DATA, default: see CRANE below)."""

import os

import pytest
import torch
from PIL import Image

import autolabel.models as models
from autolabel import pipeline
from autolabel.models import detector_modes, get_detector
from autolabel.models.base import ClassSpec, Example, Prediction
from autolabel.models.owlv2 import (Owlv2ExampleRun, Owlv2TextDetector, class_threshold,
                                    example_box_index, pos_value)
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


# ---- owlv2 (CONTRACT #9 owlv2 rules: cosine, max over examples, rule C threshold) ----

class FakeOwl:
    """scenes: what box_features returns, one (embeds, boxes, objectness) per call, in order."""

    def __init__(self, scenes=()):
        self.scenes = list(scenes)
        self.calls = 0

    def box_features(self, image):
        self.calls += 1
        return self.scenes.pop(0)


def scene(embeds, boxes=None, objectness=None):
    """embeds: one vector per predicted box (normalized here). boxes: pixels x, y, w, h on a 100 px
    image; default one far-apart 8 px box each, so NMS leaves them alone. objectness default 1."""
    boxes = boxes or [(i % 10 * 10, i // 10 * 10, 8, 8) for i in range(len(embeds))]
    rel = [((x + w / 2) / 100, (y + h / 2) / 100, w / 100, h / 100) for x, y, w, h in boxes]
    obj = objectness or [1.0] * len(embeds)
    e = torch.nn.functional.normalize(torch.tensor(embeds, dtype=torch.float32), dim=-1)
    return e, torch.tensor(rel, dtype=torch.float32), torch.tensor(obj, dtype=torch.float32)


def owl_run(classes, examples, thresholds, scenes, floors=None):
    """A run with chosen example embeddings and thresholds, to test detect on its own."""
    ex_t = [torch.nn.functional.normalize(torch.tensor(e, dtype=torch.float32), dim=-1) for e in examples]
    return Owlv2ExampleRun(FakeOwl(scenes), classes, ex_t, thresholds,
                           floors or [0.0] * len(classes), {c.name: 5 for c in classes})


IM100 = Image.new("RGB", (100, 100))
PERSON, BIAX = ClassSpec("person", "p"), ClassSpec("biax", "b")
FIVE = [(i * 20, 0, 10, 10) for i in range(5)]             # five far-apart example boxes


def five(path, category="person"):
    return [ex(path, category, b) for b in FIVE]


def test_owlv2_example_embedding_picks_highest_objectness_among_near_best_iou(img):
    target = (0, 0, 40, 40)
    boxes = [(0, 0, 40, 40),      # IoU 1.0,  objectness 0.2
             (2, 0, 40, 40),      # IoU 0.905 (>= 0.9 x best), objectness 0.9  <- picked
             (20, 0, 40, 40),     # IoU 0.33, objectness 0.99: too far from the best IoU
             (60, 60, 8, 8)]
    s = scene([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], boxes, [0.2, 0.9, 0.99, 0.5])
    assert example_box_index(s[1], s[2], [0, 0, 0.4, 0.4]) == 1
    run = Owlv2TextDetector(backend=FakeOwl([s])).prepare_examples(
        [PERSON], {"person": [ex(img, "person", target)] * 5})
    assert run.examples[0].tolist() == [[0.0, 1.0, 0.0]] * 5


def test_owlv2_score_is_raw_cosine_max_over_examples_not_mean():
    # Examples [1, 0] and [0, 1]. The mean prototype would give 0.707 to [1, 0]; max gives 1.0.
    run = owl_run([PERSON], [[[1, 0], [0, 1]]], [0.0], [scene([[1, 0], [0.6, 0.8], [-1, 0]])])
    preds = run.detect(IM100, "7")
    assert [p.score for p in preds] == pytest.approx([1.0, 0.8, 0.0])  # [-1, 0]: best is 0 (vs [0, 1])
    assert all(p.source == "owlv2_image" and p.image_id == "7" and p.polygon is None for p in preds)


def test_owlv2_objectness_filter_is_0_01():
    run = owl_run([PERSON], [[[1, 0]]], [0.0], [scene([[1, 0], [1, 0]], objectness=[0.009, 0.01])])
    preds = run.detect(IM100, "1")
    assert [p.bbox[0] for p in preds] == [10]          # only the 0.01 box (the second, at x 10)


@pytest.mark.parametrize("shift,kept", [(20, 1), (27, 2)])   # IoU 0.33 -> suppressed, IoU 0.19 -> kept
def test_owlv2_nms_is_class_agnostic_at_0_3(shift, kept):
    boxes = [(0, 0, 40, 40), (shift, 0, 40, 40)]
    run = owl_run([PERSON, BIAX], [[[1, 0]], [[0, 1]]], [0.5, 0.5], [scene([[1, 0], [0.1, 1]], boxes)])
    preds = run.detect(IM100, "1")
    assert len(preds) == kept and preds[0].category == "person"
    assert [p.category for p in preds] == ["person", "biax"][:kept]


def test_owlv2_box_passing_two_classes_keeps_the_higher_score():
    run = owl_run([PERSON, BIAX], [[[1, 0]], [[0, 1]]], [0.0, 0.0], [scene([[0.8, 0.6]])])
    assert [(p.category, round(p.score, 3)) for p in run.detect(IM100, "1")] == [("person", 0.8)]


def test_owlv2_box_must_pass_its_class_threshold_and_floor():
    run = owl_run([PERSON, BIAX], [[[1, 0]], [[0, 1]]], [0.9, 0.1],
                  [scene([[0.8, 0.6], [0.6, 0.8], [0.95, 0.31]])], floors=[0.05, 0.7])
    # box 0: person 0.8 < 0.9, biax 0.6 < floor 0.7; box 1: biax 0.8 passes; box 2: person 0.95 passes.
    assert [(p.category, round(p.score, 2)) for p in run.detect(IM100, "1")] == [("person", 0.95), ("biax", 0.8)]


@pytest.mark.parametrize("pos,bg,expected", [(0.9, 0.8, 0.85), (0.8, 0.9, 0.8), (0.5, None, 0.5), (-0.2, -0.4, 0.0)])
def test_owlv2_threshold_is_halfway_never_above_pos(pos, bg, expected):
    assert class_threshold(pos, bg) == pytest.approx(expected)


def test_owlv2_pos_is_median_leave_one_out_cosine():
    e = torch.nn.functional.normalize(torch.tensor(
        [[1, 0, 0], [1, 0, 0], [0.8, 0.6, 0], [0.6, 0.8, 0], [0, 1, 0]], dtype=torch.float32), dim=-1)
    assert pos_value(e) == pytest.approx(0.96)          # best others: 1, 1, 0.96, 0.96, 0.8
    assert pos_value(e[1:]) == pytest.approx(0.88)      # even count: 0.8, 0.96, 0.96, 0.8 -> mean of 0.8 and 0.96


def test_owlv2_prepare_threshold_from_pos_and_bg(img):
    # One image holds the 5 examples (pos 0.96, see above) plus three other boxes:
    #  [0.6, 0, 0.8] free background, score 0.6 -> bg 0.6
    #  [1, 0, 0] with objectness 0.005: ignored by the objectness filter
    #  [1, 0, 0] overlapping an example (IoU 0.6 >= 0.3): not background
    embeds = [[1, 0, 0], [1, 0, 0], [0.8, 0.6, 0], [0.6, 0.8, 0], [0, 1, 0], [0.6, 0, 0.8], [1, 0, 0], [1, 0, 0]]
    boxes = FIVE + [(50, 50, 10, 10), (70, 70, 10, 10), (0, 0, 10, 6)]
    s = scene(embeds, boxes, [1.0] * 6 + [0.005, 1.0])
    backend = FakeOwl([s])
    run = Owlv2TextDetector(backend=backend).prepare_examples([PERSON], {"person": five(img)})
    assert backend.calls == 1                            # one model pass per example image
    assert run.thresholds == pytest.approx([(0.96 + 0.6) / 2])
    assert run.used == {"person": 5} and run.unusable == [] and run.skip_reasons == {}


def test_owlv2_bg_is_median_over_example_images_and_pos_without_bg(img, tmp_path):
    other = tmp_path / "other.png"
    Image.new("RGB", (100, 50)).save(other)
    same = [[1, 0]] * 5
    # image 1: 3 examples + background 0.2; image 2: 2 examples + background 0.6 -> bg = median(0.2, 0.6) = 0.4
    s1 = scene(same[:3] + [[0.2, 0.9798]], FIVE[:3] + [(60, 60, 8, 8)])
    s2 = scene(same[:2] + [[0.6, 0.8]], FIVE[:2] + [(60, 60, 8, 8)])
    items = [ex(img, "person", b) for b in FIVE[:3]] + [ex(str(other), "person", b) for b in FIVE[:2]]
    run = Owlv2TextDetector(backend=FakeOwl([s1, s2])).prepare_examples([PERSON], {"person": items})
    assert run.thresholds == pytest.approx([(1.0 + 0.4) / 2], abs=1e-4)
    # No free box anywhere -> threshold = pos.
    lonely = scene(same, FIVE)
    run = Owlv2TextDetector(backend=FakeOwl([lonely])).prepare_examples([PERSON], {"person": five(img)})
    assert run.thresholds == pytest.approx([1.0])


# bg and Example.labelled (CONTRACT owlv2 `bg`): the worker puts every NON-REJECTED box of the class on the
# example image into `labelled` (any source or status, including boxes beyond the 30 cap). The detector
# excludes bbox + labelled from bg; anything else (rejected, other class, never boxed) is background.
# Scene: 5 examples [1, 0] (pos 1.0), box U scoring 0.9 at (50, 50), free box scoring 0.3 at (70, 70).
U = (50, 50, 10, 10)


def _bg_scene():
    return scene([[1, 0]] * 5 + [[0.9, 0.43589], [0.3, 0.95394]], FIVE + [U, (70, 70, 10, 10)])


def _labelled(path, category, labelled_first=()):
    """5 examples; only the first carries `labelled` (the union over examples is what counts)."""
    return [Example("1", path, category, b, tuple(labelled_first) if i == 0 else ()) for i, b in enumerate(FIVE)]


@pytest.mark.parametrize("case", ["unreviewed (needs_review) same-class box",
                                  "imported same-class box",
                                  "same-class box beyond the 30 cap"])
def test_owlv2_bg_excludes_labelled_boxes(img, case):
    # The worker passes all three in `labelled`, so U is not background: bg = 0.3, threshold (1 + 0.3) / 2.
    run = Owlv2TextDetector(backend=FakeOwl([_bg_scene()])).prepare_examples(
        [PERSON], {"person": _labelled(img, "person", [U])})
    assert run.thresholds == pytest.approx([0.65], abs=1e-4)


def test_owlv2_bg_counts_a_rejected_box(img):
    # A rejected box is not in `labelled` (the worker leaves it out), so U is background: bg = 0.9.
    run = Owlv2TextDetector(backend=FakeOwl([_bg_scene()])).prepare_examples(
        [PERSON], {"person": _labelled(img, "person")})
    assert run.thresholds == pytest.approx([0.95], abs=1e-4)


def test_owlv2_bg_counts_other_class_boxes(img):
    # U is a labelled biax box on the same image: excluded for biax, background for person.
    run = Owlv2TextDetector(backend=FakeOwl([_bg_scene()])).prepare_examples(
        [PERSON, BIAX], {"person": _labelled(img, "person"), "biax": _labelled(img, "biax", [U])})
    assert run.thresholds == pytest.approx([0.95, 0.65], abs=1e-4)


def test_example_labelled_defaults_to_empty():
    assert Example("1", "p", "person", (1, 2, 3, 4)).labelled == ()


def test_owlv2_class_with_4_examples_is_skipped_with_reason_5_runs(img):
    s = scene([[1, 0]] * 5 + [[0, 1]], FIVE + [(60, 60, 8, 8)])
    backend = FakeOwl([s])
    run = Owlv2TextDetector(backend=backend).prepare_examples(
        CLASSES, {"person": five(img)[:4], "biax": five(img, "biax")})
    assert [c.name for c in run.classes] == ["biax"]     # sb: no examples, not listed (worker: "no examples")
    assert run.unusable == ["person"]
    assert run.skip_reasons == {"person": "needs at least 5 examples"}
    assert run.used == {"biax": 5}
    assert backend.calls == 1


def test_owlv2_no_top_5_cap():
    rows = [[1.0, 0.1 * i] for i in range(8)]
    preds = owl_run([PERSON], [[[1, 0]]], [0.0], [scene(rows)]).detect(IM100, "1")
    assert len(preds) == 8
    assert [p.score for p in preds] == sorted(p.score for p in preds)[::-1]   # best first


def test_owlv2_boxes_are_clipped_and_empty_ones_dropped():
    im = Image.new("RGB", (100, 50))                    # wide: the padded square is 100 x 100
    boxes = [(90, 40, 20, 20), (10, 70, 8, 8)]          # past the edge; fully in the padding
    preds = owl_run([PERSON], [[[1, 0]]], [0.0], [scene([[1, 0], [1, 0]], boxes)]).detect(im, "1")
    assert [p.bbox for p in preds] == [pytest.approx((90, 40, 10, 10))]


def test_owlv2_no_box_above_threshold_returns_empty():
    run = owl_run([PERSON], [[[1, 0]]], [0.9], [scene([[0, 1], [0.5, 0.5]])])
    assert run.detect(IM100, "1") == []


def test_owlv2_run_with_no_classes_finds_nothing():
    run = Owlv2TextDetector(backend=FakeOwl()).prepare_examples(CLASSES, {})
    assert run.classes == [] and run.unusable == [] and run.detect(Image.new("RGB", (8, 8)), "1") == []


def test_owlv2_text_detect_still_uses_its_own_source():
    backend = FakeOwl()
    backend.detect = lambda image, prompts, threshold: [Detection(0, 0.7, (1, 2, 3, 4))]
    preds = Owlv2TextDetector(backend=backend).detect(Image.new("RGB", (8, 8)), "1", CLASSES)
    assert [p.source for p in preds] == ["owlv2"]


def test_owlv2_text_scores_unchanged():
    # Text mode: the backend's sigmoid is the score, no threshold rule, no cap.
    scores = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    backend = FakeOwl()
    backend.detect = lambda image, prompts, threshold: [Detection(0, s, (i, 0, 3, 4)) for i, s in enumerate(scores)]
    preds = Owlv2TextDetector(backend=backend).detect(Image.new("RGB", (8, 8)), "1", CLASSES[:1])
    assert [p.score for p in preds] == pytest.approx(scores)


# Crane frames for the slow test (gitignored data; AUTOLABEL_CRANE_DATA overrides the folder).
CRANE = os.environ.get("AUTOLABEL_CRANE_DATA", r"D:\autoannotate\testdata\v1_yolo_unzipped")
RB_EXAMPLES = [   # rb boxes x1, y1, x2, y2 px, from the provisional Roboflow labels
    ("rb_1__cam0_2026-09-29_16-24-22_00000_jpg.rf.d2810ddfc9001ba14f98395ed989cb7d.jpg", (645, 151, 714, 446)),
    ("rb_1__cam0_2026-09-29_16-24-22_00599_jpg.rf.3c0294df4705ef456dfd84c4a2a80c84.jpg", (574, 152, 671, 532)),
    ("rb_1__cam0_2026-09-29_16-34-20_00320_jpg.rf.390160db9e8d6a492be3e32f5b21ea95.jpg", (557, 161, 668, 552)),
    ("rb_1__cam0_2026-09-29_16-44-25_00028_jpg.rf.445f08aff4a838eb8924f98eb4376f40.jpg", (561, 191, 634, 499)),
    ("rb_load__cam0_2026-09-29_18-14-10_00237_jpg.rf.3c82467ae591ceabcbadc573644c9d9f.jpg", (517, 136, 712, 583))]
RB_TEST = ("rb_load__cam0_2026-09-29_18-24-11_00000_jpg.rf.e817a2afc43fa7115f5bae32a93bd093.jpg", (539, 153, 712, 546))


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


@pytest.mark.skipif(os.environ.get("AUTOLABEL_SLOW") != "1", reason="slow: set AUTOLABEL_SLOW=1")
def test_owlv2_real_model_finds_rb_on_a_crane_frame():
    images = os.path.join(CRANE, "train", "images")
    if not os.path.isdir(images):
        pytest.skip(f"crane frames not found in {images} (set AUTOLABEL_CRANE_DATA)")
    examples = {"rb": [Example("1", os.path.join(images, f), "rb", (x1, y1, x2 - x1, y2 - y1))
                       for f, (x1, y1, x2, y2) in RB_EXAMPLES],
                "person": [Example("1", os.path.join(images, RB_EXAMPLES[0][0]), "person", (2, 215, 53, 39))] * 4}
    det = Owlv2TextDetector(device="cpu")
    run = det.prepare_examples([ClassSpec("person", "person"), ClassSpec("rb", "rb")], examples)
    assert run.unusable == ["person"] and run.skip_reasons == {"person": "needs at least 5 examples"}
    assert run.used == {"rb": 5} and 0 < run.thresholds[0] <= 1

    frame = Image.open(os.path.join(images, RB_TEST[0])).convert("RGB")
    embeds, boxes, objectness = det.backend.box_features(frame)
    assert embeds.shape[0] == boxes.shape[0] == objectness.shape[0] and boxes.shape[1] == 4
    assert torch.linalg.vector_norm(embeds, dim=1) == pytest.approx(torch.ones(len(embeds)), abs=1e-4)
    preds = run.detect(frame, "x")
    print(f"\nOWLv2 examples on a crane frame: rb threshold {run.thresholds[0]:.3f}, "
          f"{len(preds)} boxes, scores {[round(p.score, 3) for p in preds]}")
    assert preds and all(p.category == "rb" and p.source == "owlv2_image" and 0 <= p.score <= 1 for p in preds)
    best = max(_iou((p.bbox[0], p.bbox[1], p.bbox[0] + p.bbox[2], p.bbox[1] + p.bbox[3]), RB_TEST[1]) for p in preds)
    assert best >= 0.5


def test_get_detector_fake_has_prepare_examples():
    assert callable(get_detector("fake").prepare_examples)
