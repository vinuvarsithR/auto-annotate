"""
postprocess.py - OWLv2 + SAM 2 steps 1 and 2 (docs/CONTRACT.md "OWLv2 + SAM 2 polygons").

Plain-English summary
---------------------
* merge_contained (step 1): two boxes of the same class where the smaller one is mostly (>= 80%)
  inside the other become one box: the union, with the higher score.
* hook_filter (step 2): the camera rides on the crane, so a hooked load hangs near a fixed image
  point (the video's hook point). Load boxes whose center is farther than `radius` px from it are
  dropped, then only the best-scoring load box is kept (one load per image). Persons are untouched.

Pure Python: no torch, no numpy. Boxes are Prediction objects, bbox = (x, y, w, h) px.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Iterable

from autolabel.models.base import Prediction

CONTAIN_FRAC = 0.8


def _area(b: tuple) -> float:
    return b[2] * b[3]


def _intersection(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return ix * iy


def _union_box(a: tuple, b: tuple) -> tuple:
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return (x0, y0, x1 - x0, y1 - y0)


def _contained_pair(preds: list[Prediction], frac: float) -> tuple[int, int] | None:
    """First (i, j) in contract scan order: j is the smaller-or-equal box of the same class,
    with >= frac of its area inside box i."""
    for i, big in enumerate(preds):
        for j, small in enumerate(preds):
            if j == i or small.category != big.category:
                continue
            a = _area(small.bbox)
            if a <= _area(big.bbox) and a > 0 and _intersection(big.bbox, small.bbox) >= frac * a:
                return i, j
    return None


def merge_contained(preds: list[Prediction], frac: float = CONTAIN_FRAC) -> list[Prediction]:
    """Step 1. Merged box = union of the two bboxes, score = the higher one, every other field
    from the larger box; it takes the larger box's place, the smaller one is removed. Repeats until
    no pair is left. Different classes never merge."""
    out = list(preds)
    while (pair := _contained_pair(out, frac)) is not None:
        i, j = pair
        big, small = out[i], out[j]
        out[i] = replace(big, bbox=_union_box(big.bbox, small.bbox), score=max(big.score, small.score))
        del out[j]
    return out


def _center_distance(bbox: tuple, point: tuple[float, float]) -> float:
    return math.hypot(bbox[0] + bbox[2] / 2 - point[0], bbox[1] + bbox[3] / 2 - point[1])


def hook_filter(preds: list[Prediction], hook: tuple[float, float] | None,
                load_classes: Iterable[str], radius: float) -> list[Prediction]:
    """Step 2. No hook point, or no load classes: returned unchanged.

    Load box with center distance > radius: dropped (<= radius kept). Then one load per image: only
    the load box with the highest score stays (first in list order on a tie), all load classes
    together. Order of what is kept is unchanged."""
    loads = set(load_classes)
    if hook is None or not loads:
        return list(preds)
    near = [p for p in preds if p.category not in loads or _center_distance(p.bbox, hook) <= radius]
    best = None
    for p in near:
        if p.category in loads and (best is None or p.score > best.score):
            best = p
    return [p for p in near if p.category not in loads or p is best]
