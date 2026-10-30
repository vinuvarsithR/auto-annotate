"""Tests for the GPU-free helpers in autolabel/owlv2.py (image-guided queries)."""

import pytest
import torch

from autolabel.owlv2 import best_box_index, class_max, xywh_to_rel_xyxy


def test_xywh_to_rel_xyxy_uses_longer_side():
    # 1280x720 image is padded to 1280x1280
    assert xywh_to_rel_xyxy((640, 360, 128, 64), 1280) == [0.5, 0.28125, 0.6, 0.33125]


def test_best_box_index_picks_most_overlapping():
    preds = [[0.1, 0.1, 0.1, 0.1], [0.55, 0.55, 0.1, 0.1], [0.5, 0.5, 0.5, 0.5]]  # cx, cy, w, h
    i, overlap = best_box_index(preds, [0.5, 0.5, 0.6, 0.6])
    assert i == 1
    assert overlap == pytest.approx(1.0)


def test_class_max_keeps_best_example_per_class():
    logits = torch.tensor([[1.0, 3.0, -1.0], [2.0, 0.0, 5.0]])  # 2 patches, 3 queries
    out = class_max(logits, owner=[0, 0, 1], n_classes=2)
    assert out.tolist() == [[3.0, -1.0], [2.0, 5.0]]


def test_class_max_class_without_examples_is_minus_inf():
    out = class_max(torch.zeros(1, 1), owner=[0], n_classes=2)
    assert out[0, 1] == float("-inf")
