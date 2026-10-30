"""Tests for autolabel/merge.py."""

from autolabel.evaluate import Box
from autolabel.merge import keep_higher

SB_BIAX = {"sb", "biax"}


def B(img, cls, x, s):
    return Box(img, cls, (x, 0, 100, 10), s)


def test_overlapping_sb_biax_keeps_higher():
    out = keep_higher([B("a", "sb", 0, 0.3), B("a", "biax", 2, 0.6)], SB_BIAX)
    assert [p.category for p in out] == ["biax"]


def test_non_overlapping_both_kept():
    out = keep_higher([B("a", "sb", 0, 0.3), B("a", "biax", 500, 0.6)], SB_BIAX)
    assert len(out) == 2


def test_other_classes_untouched():
    out = keep_higher([B("a", "sb", 0, 0.3), B("a", "person", 0, 0.9)], SB_BIAX)
    assert len(out) == 2


def test_same_class_overlap_untouched():
    out = keep_higher([B("a", "sb", 0, 0.3), B("a", "sb", 2, 0.6)], SB_BIAX)
    assert len(out) == 2


def test_different_images_untouched():
    out = keep_higher([B("a", "sb", 0, 0.3), B("b", "biax", 0, 0.6)], SB_BIAX)
    assert len(out) == 2


def test_dropped_box_cannot_remove_others():
    # biax 0.9 removes sb 0.5; sb 0.5 must not then remove biax 0.4 at another spot it overlaps.
    preds = [Box("a", "biax", (0, 0, 100, 10), 0.9), Box("a", "sb", (5, 0, 100, 10), 0.5),
             Box("a", "biax", (10, 0, 100, 10), 0.4)]
    out = keep_higher(preds, SB_BIAX)
    assert sorted(p.score for p in out) == [0.4, 0.9]
