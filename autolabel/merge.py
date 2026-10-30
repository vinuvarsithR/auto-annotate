"""
merge.py - resolve boxes where two look-alike classes claim the same object.

Image-guided OWLv2 scores every class separately, so one beam can get both an
'sb' box and a 'biax' box. keep_higher() keeps only the higher-scoring box
when boxes of the given classes overlap (IoU >= iou_threshold) in one image.
The default elsewhere is to keep both (favours recall); this is for comparing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from autolabel.evaluate import Box, iou


def keep_higher(preds: Iterable[Box], classes: set[str], iou_threshold: float = 0.5) -> list[Box]:
    preds = list(preds)
    by_img: dict[str, list[int]] = defaultdict(list)
    for i, p in enumerate(preds):
        if p.category in classes:
            by_img[p.image_id].append(i)

    dropped: set[int] = set()
    for idx in by_img.values():
        for i in sorted(idx, key=lambda k: preds[k].score, reverse=True):
            if i in dropped:
                continue
            for j in idx:
                if j != i and j not in dropped and preds[j].category != preds[i].category \
                        and preds[j].score <= preds[i].score and iou(preds[i].bbox, preds[j].bbox) >= iou_threshold:
                    dropped.add(j)
    return [p for k, p in enumerate(preds) if k not in dropped]
