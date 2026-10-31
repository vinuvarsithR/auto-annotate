"""Tests for the shared types, the registry and the "fake" detector. CPU only."""

import subprocess
import sys

import pytest
from PIL import Image

from autolabel.evaluate import ALL, Box, evaluate
from autolabel.models import get_detector, list_detectors
from autolabel.models.base import ClassSpec, Prediction
from autolabel.models.fake import FakeDetector

CLASSES = [
    ClassSpec("person", "a worker wearing a hard hat"),
    ClassSpec("biax", "a short yellow crossbar"),
    ClassSpec("rb", "a long beige slab"),
]


def _img(w=1280, h=720):
    return Image.new("RGB", (w, h))


def test_two_boxes_first_two_classes_and_scores():
    preds = FakeDetector().detect(_img(), "img7", CLASSES)
    assert [(p.category, p.score) for p in preds] == [("person", 0.9), ("biax", 0.3)]
    assert all(p.source == "fake" and p.mask is None for p in preds)
    assert all(p.image_id == "img7" for p in preds)


@pytest.mark.parametrize("w,h", [(1280, 720), (37, 91)])
def test_boxes_inside_image(w, h):
    for p in FakeDetector().detect(_img(w, h), "a", CLASSES):
        x, y, bw, bh = p.bbox
        assert bw > 0 and bh > 0
        assert x >= 0 and y >= 0 and x + bw <= w and y + bh <= h


def test_same_output_every_call():
    det = FakeDetector()
    assert det.detect(_img(), "a", CLASSES) == det.detect(_img(), "a", CLASSES)


def test_min_score_drops_low_box():
    classes = [ClassSpec("person", "p"), ClassSpec("biax", "b", min_score=0.5)]
    preds = FakeDetector().detect(_img(), "a", classes)
    assert [p.category for p in preds] == ["person"]


def test_one_class_and_no_classes():
    assert len(FakeDetector().detect(_img(), "a", CLASSES[:1])) == 1
    assert FakeDetector().detect(_img(), "a", []) == []


def test_prediction_is_a_box_and_works_with_evaluate():
    preds = FakeDetector().detect(_img(), "a", CLASSES)
    assert all(isinstance(p, Box) for p in preds)
    gold = [Box("a", "person", preds[0].bbox)]
    res = evaluate(preds, gold)
    assert res["person"]["tp"] == 1
    assert res[ALL]["fp"] == 1  # the biax box has no gold


def test_registry_lists_fake_and_caches():
    assert "fake" in list_detectors()
    d1 = get_detector("fake")
    assert d1 is get_detector("fake")
    assert d1.name == "fake"


def test_registry_unknown_name():
    with pytest.raises(ValueError, match="unknown detector"):
        get_detector("nope")


def test_no_torch_imported():
    # None of our modules may pull in torch. Checked in a fresh process,
    # because other test files import torch into this one.
    code = ("import sys, autolabel.models, autolabel.models.fake, autolabel.pipeline; "
            "sys.exit(1 if 'torch' in sys.modules else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_prediction_with_mask_is_hashable():
    """mask is excluded from hash/== (contract change, see docs/agents/models.md)."""
    rle = {"size": [1, 1], "counts": "1"}
    p = Prediction("a", "person", (0, 0, 1, 1), 0.5, source="x", mask=rle)
    hash(p)  # no TypeError
    same_but_no_mask = Prediction("a", "person", (0, 0, 1, 1), 0.5, source="x")
    assert p == same_but_no_mask
    assert len({p, same_but_no_mask}) == 1
    assert p.mask == rle  # the mask is still stored
