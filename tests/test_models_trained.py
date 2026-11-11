"""Uploaded trained weights as a detector (#24 - #26, section 3 "Trained weights").
Written by the integrator from docs/CONTRACT.md. Models are stand-ins; no weights, no GPU.

autolabel.models.trained is imported inside a fixture, so while it is missing each test fails
on its own instead of stopping the whole pytest run at collection."""

import types

import numpy as np
import pytest
from PIL import Image

from autolabel.models.base import ClassSpec

CLASSES = [ClassSpec("person", ""), ClassSpec("rb", ""), ClassSpec("sb", "", min_score=0.3),
           ClassSpec("biax", "")]


@pytest.fixture
def m():
    import autolabel.models as models
    yield models
    for name in ("t_v1", "t_v2"):
        models.unregister_trained(name)


class FakeMasks:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)


class FakeBoxes:
    def __init__(self, rows):          # rows: (x1, y1, x2, y2, conf, cls)
        a = np.array(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]


class FakeResult:
    def __init__(self, rows, masks=None):
        self.boxes = FakeBoxes(rows)
        self.masks = FakeMasks(masks) if masks is not None else None


class FakeModel:
    """Stands in for ultralytics.YOLO: .task, .names, .ckpt, .overrides, .predict()."""

    def __init__(self, task="detect", names=None, rows=(), masks=None, imgsz=960):
        self.task = task
        self.names = names if names is not None else {0: "person", 1: "rb", 2: "sb", 3: "biax"}
        self.rows, self.masks, self.calls = list(rows), masks, []
        self.ckpt = {"train_args": {"imgsz": imgsz}} if imgsz else {}
        self.overrides = {}

    def predict(self, image, **kw):
        self.calls.append(kw)
        keep = [i for i, r in enumerate(self.rows)
                if r[4] >= kw["conf"] and ("classes" not in kw or int(r[5]) in kw["classes"])]
        masks = None if self.masks is None else [self.masks[i] for i in keep]
        return [FakeResult([self.rows[i] for i in keep], masks)]


def rect_mask(h, w, x0, y0, x1, y1):
    a = np.zeros((h, w))
    a[y0:y1, x0:x1] = 1.0
    return a


# ------------------------------------------------------------------ inspect_weights

def test_inspect_reads_task_names_imgsz(m):
    info = m.inspect_weights("unused.pt", model=FakeModel("segment", {1: "rb", 0: "person"}))
    assert info.task == "segment"
    assert info.class_names == ["person", "rb"]  # index order
    assert info.imgsz == 960


def test_inspect_list_names_and_no_imgsz(m):
    info = m.inspect_weights("unused.pt", model=FakeModel("pose", ["a", "b"], imgsz=None))
    assert (info.task, info.class_names, info.imgsz) == ("pose", ["a", "b"], None)


def test_inspect_garbage_file_is_one_line_value_error(m, tmp_path):
    bad = tmp_path / "best.pt"
    bad.write_bytes(b"this is not a checkpoint")
    with pytest.raises(ValueError) as e:
        m.inspect_weights(str(bad))
    assert "\n" not in str(e.value)


def test_inspect_missing_file_never_downloads(m, tmp_path):
    with pytest.raises(ValueError):
        m.inspect_weights(str(tmp_path / "nothing" / "yolo11n.pt"))
    assert not (tmp_path / "nothing").exists()


# ------------------------------------------------------------------ registry

def test_registered_name_works_but_is_not_listed(m, tmp_path):
    fake = FakeModel()
    m.register_trained("t_v1", str(tmp_path / "t_v1.pt"), "detect", model=fake)
    assert "t_v1" not in m.list_detectors()
    assert m.detector_modes("t_v1") == []
    det = m.get_detector("t_v1")
    assert det.detect is not None


def test_same_path_keeps_instance_new_path_replaces(m, tmp_path):
    m.register_trained("t_v1", str(tmp_path / "a.pt"), "detect", model=FakeModel())
    first = m.get_detector("t_v1")
    m.register_trained("t_v1", str(tmp_path / "a.pt"), "detect", model=FakeModel())
    assert m.get_detector("t_v1") is first
    m.register_trained("t_v1", str(tmp_path / "b.pt"), "detect", model=FakeModel())
    assert m.get_detector("t_v1") is not first


def test_unregister(m, tmp_path):
    m.register_trained("t_v1", str(tmp_path / "a.pt"), "detect", model=FakeModel())
    m.get_detector("t_v1")
    m.unregister_trained("t_v1")
    with pytest.raises(ValueError):
        m.get_detector("t_v1")
    m.unregister_trained("never_registered")  # no-op


# ------------------------------------------------------------------ detection

def test_detect_task_boxes_by_name_and_floors(m, tmp_path):
    names = {0: "biax", 1: "person", 2: "beam", 3: "sb"}  # model order differs; "beam" unknown
    rows = [(10, 20, 110, 70, 0.9, 1),   # person
            (0, 0, 50, 50, 0.9, 2),      # beam: not a project class -> ignored
            (5, 5, 25, 15, 0.2, 3),      # sb below its 0.3 floor -> dropped
            (30, 30, 60, 90, 0.04, 0)]   # biax below the 0.05 default -> dropped
    m.register_trained("t_v1", str(tmp_path / "a.pt"), "detect",
                       model=FakeModel("detect", names, rows))
    preds = m.get_detector("t_v1").detect(Image.new("RGB", (200, 100)), "7", CLASSES)
    assert [(p.category, p.bbox, p.source, p.polygon, p.mask) for p in preds] == [
        ("person", (10.0, 20.0, 100.0, 50.0), "yolo_trained", None, None)]
    assert preds[0].image_id == "7" and preds[0].score == pytest.approx(0.9)


def test_imgsz_from_checkpoint(m, tmp_path):
    fake = FakeModel(rows=[(10, 20, 110, 70, 0.9, 0)], imgsz=960)
    m.register_trained("t_v1", str(tmp_path / "a.pt"), "detect", model=fake)
    m.get_detector("t_v1").detect(Image.new("RGB", (200, 100)), "1", CLASSES)
    assert fake.calls[0]["imgsz"] == 960


def test_segment_task_sets_polygon_from_mask(m, tmp_path):
    h, w = 100, 200
    rows = [(20, 10, 120, 60, 0.9, 1)]  # rb
    masks = [rect_mask(h, w, 40, 20, 100, 50)]  # tighter than the model box
    fake = FakeModel("segment", rows=rows, masks=masks)
    m.register_trained("t_v2", str(tmp_path / "s.pt"), "segment", model=fake)
    (p,) = m.get_detector("t_v2").detect(Image.new("RGB", (w, h)), "1", CLASSES)
    assert fake.calls[0].get("retina_masks") is True
    assert p.category == "rb" and p.source == "yolo_trained" and p.mask is None
    assert p.polygon is not None and len(p.polygon) >= 3
    xs = [pt[0] * w for pt in p.polygon]
    ys = [pt[1] * h for pt in p.polygon]
    assert p.bbox == pytest.approx((min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)),
                                   abs=1e-3)
    assert p.bbox[0] == pytest.approx(40, abs=1.5) and p.bbox[2] == pytest.approx(60, abs=2)


def test_segment_split_mask_keeps_model_box(m, tmp_path):
    h, w = 100, 200
    rows = [(10, 10, 190, 40, 0.9, 2)]  # sb
    two = rect_mask(h, w, 10, 10, 90, 40) + rect_mask(h, w, 110, 10, 190, 40)  # two equal parts
    m.register_trained("t_v2", str(tmp_path / "s.pt"), "segment",
                       model=FakeModel("segment", rows=rows, masks=[two]))
    (p,) = m.get_detector("t_v2").detect(Image.new("RGB", (w, h)), "1", CLASSES)
    assert p.polygon is None
    assert p.bbox == pytest.approx((10.0, 10.0, 180.0, 30.0))
