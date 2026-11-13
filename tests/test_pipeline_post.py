"""pipeline.run(post=..., segmenter=...): OWLv2 + SAM 2 steps 1-4, with fake detectors and FakeSegmenter. CPU only."""

import pytest
from PIL import Image

from autolabel import pipeline
from autolabel.models.base import ClassSpec, PostProcess, Prediction
from autolabel.models.sam import FakeSegmenter

CLASSES = [ClassSpec("person", "p"), ClassSpec("rb", "r")]


@pytest.fixture
def images(tmp_path):
    out = []
    for i in range(2):
        path = tmp_path / f"im{i}.png"
        Image.new("RGB", (200, 100)).save(path)
        out.append((f"id{i}", str(path)))
    return out


class StubRun:
    """An ExampleRun that returns fixed boxes, so the step order can be seen."""
    classes = CLASSES
    used = {"person": 5, "rb": 5}
    unusable: list = []

    def __init__(self, boxes):
        self.boxes = boxes

    def detect(self, image, image_id):
        return [Prediction(image_id, c, b, s, source="owlv2_image") for c, b, s in self.boxes]


class RecordingSegmenter(FakeSegmenter):
    def __init__(self):
        self.calls = []

    def segment(self, image, boxes):
        self.calls.append(list(boxes))
        return super().segment(image, boxes)


def test_post_none_is_unchanged(images):
    plain = list(pipeline.run(images, CLASSES, detector="fake"))
    assert list(pipeline.run(images, CLASSES, detector="fake", post=None)) == plain


def test_fake_detector_hook_filter_by_image_id(images):
    # fake: person (20, 10, 60, 30) score 0.9, rb (100, 50, 60, 30) score 0.3; rb center (130, 65)
    post = PostProcess(load_classes=("rb",), radius=10, hooks={"id0": (0, 0)})
    out = dict(pipeline.run(images, CLASSES, detector="fake", post=post))
    assert [p.category for p in out["id0"]] == ["person"]          # rb too far from id0's hook
    assert [p.category for p in out["id1"]] == ["person", "rb"]    # id1 has no hook point: no filter
    assert all(p.polygon is None for ps in out.values() for p in ps)  # masks "none"


def test_fake_detector_with_masks_gets_polygons(images):
    post = PostProcess(masks="sam2")
    out = dict(pipeline.run(images, CLASSES, detector="fake", post=post, segmenter=FakeSegmenter()))
    for ps in out.values():
        assert [p.category for p in ps] == ["person", "rb"]
        assert all(p.polygon and len(p.polygon) >= 3 and p.mask is None for p in ps)


def test_step_order_merge_then_hook_then_masks(images):
    # rb A (score 0.5) contains rb B (score 0.9); a third rb C is far from the hook.
    # merge first -> A's box with score 0.9; hook: C dropped; one load: the merged box.
    # (hook first would keep B's small box: one load = highest score.)
    boxes = [("rb", (60.0, 20.0, 80.0, 60.0), 0.5), ("rb", (80.0, 30.0, 20.0, 20.0), 0.9),
             ("rb", (0.0, 0.0, 20.0, 20.0), 0.95), ("person", (150.0, 70.0, 20.0, 20.0), 0.4)]
    seg = RecordingSegmenter()
    post = PostProcess(load_classes=("rb",), radius=30, hooks={"id0": (100.0, 50.0), "id1": (100.0, 50.0)},
                       masks="sam2")
    out = dict(pipeline.run(images[:1], CLASSES, example_run=StubRun(boxes), post=post, segmenter=seg))
    ps = out["id0"]
    assert [(p.category, p.score) for p in ps] == [("rb", 0.9), ("person", 0.4)]
    assert seg.calls == [[(60.0, 20.0, 80.0, 60.0), (150.0, 70.0, 20.0, 20.0)]]  # only kept boxes segmented
    assert ps[0].bbox == pytest.approx((60, 20, 80, 60), abs=1.5)                  # polygon box ~ merged box


def test_missing_sam_weights_fail_before_the_first_image(images, monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOLABEL_SAM_WEIGHTS", str(tmp_path / "missing.pt"))
    seen = []
    run = pipeline.run(images, CLASSES, detector="fake", post=PostProcess(masks="sam2"),
                       on_progress=lambda d, t: seen.append(d))
    with pytest.raises(FileNotFoundError):
        next(run)
    assert seen == []


def test_masks_none_never_loads_sam(images, monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOLABEL_SAM_WEIGHTS", str(tmp_path / "missing.pt"))
    assert len(list(pipeline.run(images, CLASSES, detector="fake", post=PostProcess()))) == 2
