"""Tests for pipeline.drop_overlapping() (#9 overlap rule). CPU only, no images."""

from autolabel.evaluate import Box
from autolabel.models.base import Prediction
from autolabel.pipeline import drop_overlapping


def pred(category, bbox, score=0.8):
    return Prediction("img1", category, bbox, score, source="fake")


def test_same_class_overlap_is_dropped():
    preds = [pred("person", (0, 0, 10, 10))]
    existing = [Box("img1", "person", (1, 0, 10, 10))]   # IoU 90/110 = 0.82
    assert drop_overlapping(preds, existing) == []


def test_threshold_is_inclusive_and_low_overlap_kept():
    exact = pred("person", (0, 0, 10, 10))
    far = pred("person", (8, 0, 10, 10))                  # IoU 20/180 = 0.11
    existing = [Box("img1", "person", (0, 0, 10, 20))]   # IoU with exact = 100/200 = 0.5
    assert drop_overlapping([exact, far], existing) == [far]


def test_different_class_is_kept():
    preds = [pred("biax", (0, 0, 10, 10))]
    existing = [Box("img1", "sb", (0, 0, 10, 10))]
    assert drop_overlapping(preds, existing) == preds


def test_rejected_existing_box_does_not_block():
    # The caller passes only non-rejected boxes; a rejected box at the same spot is left out.
    preds = [pred("rb", (0, 0, 10, 10))]
    rejected = Box("img1", "rb", (0, 0, 10, 10))
    non_rejected = [b for b, status in [(rejected, "rejected")] if status != "rejected"]
    assert drop_overlapping(preds, non_rejected) == preds


def test_imported_box_blocks_a_duplicate():
    preds = [pred("sb", (5, 5, 20, 4)), pred("person", (50, 50, 10, 10))]
    imported = Prediction("img1", "sb", (5, 5, 20, 4), 1.0, source="import")
    assert drop_overlapping(preds, [imported]) == [preds[1]]


def test_order_kept_and_no_existing_keeps_all():
    preds = [pred("person", (i * 20, 0, 10, 10)) for i in range(4)]
    assert drop_overlapping(preds, []) == preds
    blocker = Box("img1", "person", (20, 0, 10, 10))
    assert drop_overlapping(preds, [blocker]) == [preds[0], preds[2], preds[3]]
