"""Tests for the "yoloe" detector. The model is mocked, except one slow test.

Slow test: skipped unless AUTOLABEL_SLOW=1. It needs AUTOLABEL_YOLOE_WEIGHTS (or
models/yoloe-26s-seg.pt) and runs on CPU on ultralytics' bundled bus.jpg. No downloads.
"""

import os

os.environ["YOLO_AUTOINSTALL"] = "false"   # set at collection: a slow test imports ultralytics, then
                                           # any unreadable-image test would pip-install pi-heif
import numpy as np
import pytest
import torch
from PIL import Image

import autolabel.models as models
from autolabel import pipeline
from autolabel.models import detector_modes, get_detector, list_detectors
from autolabel.models.base import ClassSpec, Example
from autolabel.models.yoloe import (DEFAULT_WEIGHTS, ENV_WEIGHTS, SOURCE, YoloeDetector,
                                    mask_to_polygon, polygon_bbox, resolve_weights, usable)


# ---- mask_to_polygon ----

def blank(h=100, w=200):
    return np.zeros((h, w), dtype=np.uint8)


def test_one_part_is_used():
    m = blank()
    m[20:40, 50:150] = 1
    poly = mask_to_polygon(m)
    assert len(poly) == 4
    assert polygon_bbox(poly, 200, 100) == pytest.approx((50, 20, 99, 19))


def test_output_is_normalized_and_rounded_to_6_decimals():
    m = blank(720, 1280)
    m[11:333, 17:777] = 1
    poly = mask_to_polygon(m)
    for x, y in poly:
        assert 0 <= x <= 1 and 0 <= y <= 1
        assert round(x, 6) == x and round(y, 6) == y


def test_several_parts_largest_holds_at_least_90_percent():
    m = blank()
    m[10:60, 10:110] = 1     # about 49 x 99
    m[80:83, 150:153] = 1    # tiny
    poly = mask_to_polygon(m)
    assert polygon_bbox(poly, 200, 100) == pytest.approx((10, 10, 99, 49))  # the small part is dropped


def test_several_parts_largest_under_90_percent_gives_none():
    m = blank()
    m[10:30, 10:60] = 1
    m[10:30, 100:140] = 1    # 39 x 19 against 49 x 19: largest holds about 56%
    assert mask_to_polygon(m) is None


def test_exactly_90_percent_is_kept():
    # contourArea of an N x M rectangle of pixels is (N - 1) * (M - 1)
    m = blank()
    m[0:10, 0:10] = 1        # area 81
    m[50:52, 100:110] = 1    # area 9: the largest holds exactly 90%
    assert mask_to_polygon(m) is not None


@pytest.mark.parametrize("draw", [
    lambda m: None,                                     # empty mask
    lambda m: m.__setitem__((50, 50), 1),               # one pixel: 1 point
    lambda m: m.__setitem__((50, slice(20, 90)), 1),    # a line: 2 points, zero area
])
def test_under_3_points_or_zero_area_gives_none(draw):
    m = blank()
    draw(m)
    assert mask_to_polygon(m) is None


def test_hole_is_dropped():
    m = blank()
    m[10:90, 10:190] = 1
    m[40:60, 90:110] = 0
    assert len(mask_to_polygon(m)) == 4


def test_tolerance_is_0_001_of_the_longest_side():
    # A 1-pixel bump on a long edge: within 1.28 px at 1280, so it is simplified away.
    m = blank(720, 1280)
    m[100:200, 100:1100] = 1
    m[99, 600:610] = 1
    assert len(mask_to_polygon(m)) == 4


def test_accepts_float_masks():
    m = np.zeros((50, 50), dtype=np.float32)
    m[10:30, 10:30] = 1.0
    assert mask_to_polygon(m) is not None


# ---- weights ----

def test_weights_order(monkeypatch, tmp_path):
    a, b = tmp_path / "a-seg.pt", tmp_path / "b-seg.pt"
    a.touch()
    b.touch()
    monkeypatch.setenv(ENV_WEIGHTS, str(b))
    assert resolve_weights() == str(b)
    assert resolve_weights(str(a)) == str(a)


def test_default_weights_path(monkeypatch):
    monkeypatch.delenv(ENV_WEIGHTS, raising=False)
    assert DEFAULT_WEIGHTS.parent.name == "models" and DEFAULT_WEIGHTS.name == "yoloe-26s-seg.pt"
    if not DEFAULT_WEIGHTS.exists():
        with pytest.raises(FileNotFoundError, match="yoloe weights not found"):
            resolve_weights()


def test_missing_weights_is_one_line_and_never_loads(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_WEIGHTS, str(tmp_path / "nope-seg.pt"))
    def loader(*a):
        raise AssertionError("must not load")
    with pytest.raises(FileNotFoundError) as e:
        YoloeDetector(loader=loader).prepare_examples(CLASSES, {})
    assert "\n" not in str(e.value)


@pytest.mark.parametrize("name", ["yoloe-26s-seg-pf.pt", "yoloe-26s.pt"])
def test_only_seg_files_are_accepted(tmp_path, name):
    path = tmp_path / name
    path.touch()
    with pytest.raises(ValueError, match="-seg"):
        resolve_weights(str(path))


# ---- small examples ----

def test_usable_is_measured_at_640():
    assert usable((0, 0, 100, 16), (1280, 720))        # 8 px at 640
    assert not usable((0, 0, 100, 15.9), (1280, 720))
    assert not usable((0, 0, 15, 100), (1280, 720))
    assert usable((0, 0, 8, 8), (640, 480))


# ---- prepare_examples and detect, with a fake model ----

CLASSES = [ClassSpec("person", "p"), ClassSpec("rb", "r", min_score=0.5), ClassSpec("sb", "s")]


class FakeBoxes:
    def __init__(self, rows):          # (x1, y1, x2, y2, conf, cls)
        a = np.array(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]


class FakeMasks:
    def __init__(self, data):
        self.data = data


class FakeResult:
    def __init__(self, rows, masks=None):
        self.boxes = FakeBoxes(rows)
        self.masks = FakeMasks(masks) if masks is not None else None


class FakeModel:
    instances = []

    def __init__(self, weights, device):
        self.weights, self.device = weights, device
        self.embedded, self.names, self.vpe, self.confs = [], None, None, []
        self.result = FakeResult([])
        FakeModel.instances.append(self)

    def embed(self, image, boxes, labels):
        self.embedded.append((image.size, boxes.copy(), labels.tolist()))
        firsts = [boxes[labels == k][0] for k in range(labels.max() + 1)]
        return torch.tensor([[float(b[0]), 3.0] for b in firsts])

    def set_classes(self, names, embeddings):
        self.names, self.vpe = names, embeddings

    def predict(self, image, conf):
        self.confs.append(conf)
        return self.result


@pytest.fixture
def det(monkeypatch, tmp_path):
    weights = tmp_path / "w-seg.pt"
    weights.touch()
    monkeypatch.setenv(ENV_WEIGHTS, str(weights))
    FakeModel.instances = []
    return YoloeDetector(loader=FakeModel)


@pytest.fixture
def img(tmp_path):
    path = tmp_path / "im.png"
    Image.new("RGB", (1280, 720)).save(path)
    return str(path)


def ex(path, category, bbox=(100, 100, 200, 100), image_id="1"):
    return Example(image_id, path, category, bbox)


def test_prototype_is_the_normalized_mean_over_images(det, img, tmp_path):
    other = tmp_path / "other.png"
    Image.new("RGB", (1280, 720)).save(other)
    run = det.prepare_examples(CLASSES[:1], {"person": [ex(img, "person", (0, 0, 50, 50)),
                                                        ex(str(other), "person", (8, 0, 50, 50))]})
    assert FakeModel.instances[0].names == ["person"]
    vpe = FakeModel.instances[0].vpe
    assert vpe.shape == (1, 1, 2)                # (0, 3) and (8, 3) -> mean (4, 3) -> (0.8, 0.6)
    assert vpe[0, 0].tolist() == pytest.approx([0.8, 0.6])
    assert run.used == {"person": 2}


def test_a_class_gets_one_embedding_per_image_and_used_counts_boxes(det, img):
    run = det.prepare_examples(CLASSES[:1], {"person": [ex(img, "person", (0, 0, 50, 50)),
                                                        ex(img, "person", (8, 0, 50, 50))]})
    assert FakeModel.instances[0].embedded[0][2] == [0, 0]    # both boxes in one prompt
    assert FakeModel.instances[0].vpe[0, 0].tolist() == pytest.approx([0.0, 1.0])  # (0, 3) only
    assert run.used == {"person": 2}


def test_boxes_go_in_as_corners_and_one_image_gets_one_pass(det, img):
    det.prepare_examples(CLASSES, {"person": [ex(img, "person", (10, 20, 30, 40))],
                                   "sb": [ex(img, "sb", (50, 60, 70, 80))]})
    embedded = FakeModel.instances[0].embedded
    assert len(embedded) == 1
    assert embedded[0][0] == (1280, 720)
    assert embedded[0][1].tolist() == [[10, 20, 40, 60], [50, 60, 120, 140]]
    assert embedded[0][2] == [0, 1]              # one prompt per class


def test_only_classes_with_usable_examples_run(det, img):
    run = det.prepare_examples(CLASSES, {"person": [ex(img, "person")],
                                         "rb": [ex(img, "rb", (0, 0, 100, 4))]})   # 2 px at 640
    assert [c.name for c in run.classes] == ["person"]
    assert run.used == {"person": 1}
    assert run.unusable == ["rb"]                # sb had no examples: not unusable
    assert FakeModel.instances[0].names == ["person"]


def test_too_small_example_is_not_counted(det, img):
    run = det.prepare_examples(CLASSES[:1], {"person": [ex(img, "person"),
                                                        ex(img, "person", (0, 0, 100, 4))]})
    assert run.used == {"person": 1}


def test_all_unusable_runs_nothing(det, img):
    run = det.prepare_examples(CLASSES[:1], {"person": [ex(img, "person", (0, 0, 4, 100))]})
    assert run.classes == [] and run.used == {} and run.unusable == ["person"]
    assert FakeModel.instances[0].names is None
    assert run.detect(Image.new("RGB", (64, 64)), "1") == []


def test_classes_keep_input_order(det, img):
    run = det.prepare_examples(CLASSES, {"sb": [ex(img, "sb")], "person": [ex(img, "person")]})
    assert [c.name for c in run.classes] == ["person", "sb"]
    assert FakeModel.instances[0].names == ["person", "sb"]


def test_each_job_gets_a_fresh_model(det, img):
    a = det.prepare_examples(CLASSES[:1], {"person": [ex(img, "person")]})
    b = det.prepare_examples(CLASSES[2:], {"sb": [ex(img, "sb")]})
    assert a.model is not b.model
    assert not hasattr(det, "model")


def test_unreadable_example_image_raises(det, tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    with pytest.raises(Exception):
        det.prepare_examples(CLASSES[:1], {"person": [ex(str(bad), "person")]})


def run_with(det, img, rows, masks=None):
    run = det.prepare_examples(CLASSES, {c.name: [ex(img, c.name)] for c in CLASSES})
    run.model.result = FakeResult(rows, masks)
    return run


def test_detect_without_masks_keeps_the_model_bbox(det, img):
    run = run_with(det, img, [(10, 20, 110, 70, 0.8, 0)])
    p = run.detect(Image.new("RGB", (1280, 720)), "7")[0]
    assert (p.image_id, p.category, p.source, p.mask, p.polygon) == ("7", "person", SOURCE, None, None)
    assert p.bbox == pytest.approx((10, 20, 100, 50))
    assert p.score == pytest.approx(0.8)


def test_detect_conf_is_lowest_floor_and_class_floors_apply(det, img):
    run = run_with(det, img, [(0, 0, 9, 9, 0.3, 1), (0, 0, 9, 9, 0.6, 1), (0, 0, 9, 9, 0.3, 2),
                              (0, 0, 9, 9, 0.04, 0)])
    preds = run.detect(Image.new("RGB", (64, 64)), "1")
    assert run.model.confs == [0.05]             # default floor < rb's 0.5
    assert sorted((p.category, p.score) for p in preds) == [("rb", pytest.approx(0.6)),
                                                            ("sb", pytest.approx(0.3))]


def test_detect_with_a_mask_uses_the_polygon_bbox(det, img):
    mask = np.zeros((720, 1280), dtype=np.float32)
    mask[100:200, 300:500] = 1
    run = run_with(det, img, [(1, 2, 3, 4, 0.9, 0)], masks=torch.tensor(mask[None]))
    p = run.detect(Image.new("RGB", (1280, 720)), "1")[0]
    assert len(p.polygon) == 4
    assert p.bbox == pytest.approx((300, 100, 199, 99), abs=1e-3)


def test_detect_with_a_split_mask_keeps_the_model_bbox(det, img):
    mask = np.zeros((720, 1280), dtype=np.float32)
    mask[100:120, 100:300] = 1
    mask[100:120, 400:600] = 1
    run = run_with(det, img, [(100, 100, 600, 120, 0.9, 0)], masks=torch.tensor(mask[None]))
    p = run.detect(Image.new("RGB", (1280, 720)), "1")[0]
    assert p.polygon is None
    assert p.bbox == pytest.approx((100, 100, 500, 20))


def test_score_is_clamped(det, img):
    run = run_with(det, img, [(0, 0, 9, 9, 1.2, 0)])
    assert run.detect(Image.new("RGB", (64, 64)), "1")[0].score == 1.0


def test_detect_without_examples_is_an_error(det):
    with pytest.raises(RuntimeError, match="no text mode"):
        det.detect(Image.new("RGB", (64, 64)), "1", CLASSES)


# ---- registry ----

def test_registry_is_opt_in(monkeypatch):
    monkeypatch.setenv("AUTOLABEL_DETECTORS", "")
    assert "yoloe" not in list_detectors()
    monkeypatch.setenv("AUTOLABEL_DETECTORS", "owlv2,yolo,yoloe")
    assert list_detectors() == ["fake", "owlv2", "yolo", "yoloe"]


def test_yoloe_modes():
    assert detector_modes("yoloe") == ["examples"]


# ---- the real model ----

@pytest.mark.skipif(os.environ.get("AUTOLABEL_SLOW") != "1", reason="set AUTOLABEL_SLOW=1")
def test_classes_in_one_pass_match_separate_passes():
    from ultralytics.utils import ASSETS, checks
    from autolabel.models.yoloe import UltralyticsYoloe

    with Image.open(ASSETS / "bus.jpg") as im:
        rgb = im.convert("RGB")
    boxes = np.array([[17, 230, 803, 740], [48, 398, 245, 902], [221, 405, 345, 860]], dtype=np.float32)
    model = UltralyticsYoloe(resolve_weights(), "cpu")
    assert checks.AUTOINSTALL is False
    together = model.embed(rgb, boxes, np.array([0, 1, 1]))
    bus = model.embed(rgb, boxes[:1], np.array([0]))
    people = model.embed(rgb, boxes[1:], np.array([0, 0]))
    assert together.shape[0] == 2
    assert torch.cosine_similarity(together[0], bus[0], dim=0) > 0.999
    assert torch.cosine_similarity(together[1], people[0], dim=0) > 0.999


@pytest.mark.skipif(os.environ.get("AUTOLABEL_SLOW") != "1", reason="set AUTOLABEL_SLOW=1")
def test_two_jobs_with_different_classes_in_one_process(monkeypatch):
    from ultralytics.utils import ASSETS

    monkeypatch.setenv("AUTOLABEL_DETECTORS", "yoloe")
    monkeypatch.setattr(models, "_CACHE", {})
    resolve_weights()   # stop here with a clear error when the weights are missing
    path = str(ASSETS / "bus.jpg")
    with Image.open(path) as im:
        rgb = im.convert("RGB")

    bus = ClassSpec("bus", "", min_score=0.1)
    person = ClassSpec("person", "", min_score=0.1)
    bus_box = (17.0, 230.0, 786.0, 510.0)
    person_box = (48.0, 398.0, 197.0, 504.0)
    job_a = ([bus, person], {"bus": [Example("1", path, "bus", bus_box)],
                             "person": [Example("1", path, "person", person_box)]})
    job_b = ([person], {"person": [Example("1", path, "person", person_box)]})

    run_a = pipeline.prepare_examples("yoloe", *job_a)
    first_a = run_a.detect(rgb, "1")
    run_b = pipeline.prepare_examples("yoloe", *job_b)
    first_b = run_b.detect(rgb, "1")
    second_a = run_a.detect(rgb, "1")     # job B must not have changed job A's model

    assert get_detector("yoloe") is get_detector("yoloe")
    assert run_a.model is not run_b.model
    assert run_a.used == {"bus": 1, "person": 1} and run_b.used == {"person": 1}
    assert {p.category for p in first_a} == {"bus", "person"}
    assert {p.category for p in first_b} == {"person"}
    assert [(p.category, p.bbox, p.score) for p in first_a] == [(p.category, p.bbox, p.score) for p in second_a]
    assert all(p.source == SOURCE and p.mask is None for p in first_a + first_b)
    with_polygon = [p for p in first_a + first_b if p.polygon]
    assert with_polygon
    width, height = rgb.size
    for p in with_polygon:
        assert len(p.polygon) >= 3 and all(0 <= v <= 1 for point in p.polygon for v in point)
        assert p.bbox == pytest.approx(polygon_bbox(p.polygon, width, height))

    # through pipeline.run too, on a fresh job: same boxes as calling the run directly
    again = dict(pipeline.run([("1", path)], [], detector="unused", example_run=run_b))
    assert [(p.bbox, p.score) for p in again["1"]] == [(p.bbox, p.score) for p in first_b]
