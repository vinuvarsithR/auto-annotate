"""
evaluate.py - score predicted boxes against gold boxes.

Plain-English summary
---------------------
* A box is [x, y, width, height] in pixels, top-left origin (COCO style).
* IoU ("intersection over union") = overlap area / combined area, from 0 to 1.
* A prediction is a TRUE POSITIVE if it overlaps an unclaimed gold box of the
  same class with IoU >= iou_threshold. Each gold box can be claimed once.
* Leftover predictions are FALSE POSITIVES (junk).
* Leftover gold boxes are FALSE NEGATIVES (missed).
* precision = TP / (TP + FP)   "how much of what we drew is right?"
* recall    = TP / (TP + FN)   "how much of what's there did we find?"
* class-ignored recall: same matching, but any class can claim any gold box.
  It answers "did we find the object at all, even if the class was wrong?"

No GPU or model code here: this file is pure math so it is easy to test.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

ALL = "__all__"  # key for the overall (all classes combined) row


@dataclass(frozen=True)
class Box:
    image_id: str
    category: str
    bbox: tuple[float, float, float, float]  # x, y, w, h
    score: float = 1.0  # gold boxes use 1.0


def iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two [x, y, w, h] boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _greedy_match(preds: list[Box], golds: list[Box], iou_threshold: float) -> tuple[int, int, int]:
    """
    Match predictions to gold boxes in ONE image (and one class, unless the
    caller passes mixed classes on purpose). Highest-scoring predictions pick
    first, each taking the best still-unclaimed gold box.
    Returns (tp, fp, fn).
    """
    claimed = [False] * len(golds)
    tp = 0
    for p in sorted(preds, key=lambda b: b.score, reverse=True):
        best_i, best_iou = -1, iou_threshold
        for i, g in enumerate(golds):
            if claimed[i]:
                continue
            v = iou(p.bbox, g.bbox)
            if v >= best_iou:
                best_i, best_iou = i, v
        if best_i >= 0:
            claimed[best_i] = True
            tp += 1
    fp = len(preds) - tp
    fn = len(golds) - tp
    return tp, fp, fn


def _ratio(num: int, den: int) -> Optional[float]:
    """None when undefined (e.g. precision with zero predictions)."""
    return num / den if den > 0 else None


def evaluate(
    preds: Iterable[Box],
    golds: Iterable[Box],
    score_threshold: float = 0.0,
    iou_threshold: float = 0.5,
) -> dict[str, dict]:
    """
    Score predictions against gold at one confidence threshold.

    Returns {class_name: {tp, fp, fn, n_gold, n_pred, precision, recall}}
    plus an ALL row with the totals and 'recall_class_ignored'.
    """
    preds = [p for p in preds if p.score >= score_threshold]
    golds = list(golds)

    # Group by (image, class) so boxes are only compared within one image + class.
    by_key_p: dict[tuple[str, str], list[Box]] = defaultdict(list)
    by_key_g: dict[tuple[str, str], list[Box]] = defaultdict(list)
    for p in preds:
        by_key_p[(p.image_id, p.category)].append(p)
    for g in golds:
        by_key_g[(g.image_id, g.category)].append(g)

    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for key in set(by_key_p) | set(by_key_g):
        tp, fp, fn = _greedy_match(by_key_p.get(key, []), by_key_g.get(key, []), iou_threshold)
        c = counts[key[1]]
        c["tp"] += tp
        c["fp"] += fp
        c["fn"] += fn

    results: dict[str, dict] = {}
    total = {"tp": 0, "fp": 0, "fn": 0}
    for cls in sorted(counts):
        c = counts[cls]
        for k in total:
            total[k] += c[k]
        results[cls] = _row(c)

    # Class-ignored recall: per image, any prediction may claim any gold box.
    by_img_p: dict[str, list[Box]] = defaultdict(list)
    by_img_g: dict[str, list[Box]] = defaultdict(list)
    for p in preds:
        by_img_p[p.image_id].append(p)
    for g in golds:
        by_img_g[g.image_id].append(g)
    tp_any = 0
    for img in by_img_g:
        tp, _, _ = _greedy_match(by_img_p.get(img, []), by_img_g[img], iou_threshold)
        tp_any += tp

    overall = _row(total)
    overall["recall_class_ignored"] = _ratio(tp_any, len(golds))
    results[ALL] = overall
    return results


def _row(c: dict[str, int]) -> dict:
    return {
        "tp": c["tp"],
        "fp": c["fp"],
        "fn": c["fn"],
        "n_gold": c["tp"] + c["fn"],
        "n_pred": c["tp"] + c["fp"],
        "precision": _ratio(c["tp"], c["tp"] + c["fp"]),
        "recall": _ratio(c["tp"], c["tp"] + c["fn"]),
    }


DEFAULT_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 11)]  # 0.05 .. 0.50


def sweep(
    preds: Iterable[Box],
    golds: Iterable[Box],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    iou_threshold: float = 0.5,
) -> list[dict]:
    """
    Run evaluate() at each confidence threshold.
    Returns flat rows: {threshold, class, tp, fp, fn, n_gold, n_pred, precision, recall, ...}
    """
    preds, golds = list(preds), list(golds)
    rows = []
    for t in thresholds:
        for cls, r in evaluate(preds, golds, t, iou_threshold).items():
            rows.append({"threshold": t, "class": cls, **r})
    return rows


def average_precision(preds: Iterable[Box], golds: Iterable[Box], iou_threshold: float = 0.5) -> Optional[float]:
    """
    AP for ONE class (the caller filters by class). Score-independent: it ranks
    all predictions by score, walks down the list counting hits (each gold box
    can be hit once, best-IoU unclaimed gold in the same image), and measures
    the area under the precision-recall curve (all-point interpolation:
    precision at each recall = best precision at that recall or higher).
    None when there are no gold boxes.
    """
    golds = list(golds)
    if not golds:
        return None
    gold_by_img: dict[str, list[Box]] = defaultdict(list)
    for g in golds:
        gold_by_img[g.image_id].append(g)
    claimed = {img: [False] * len(gs) for img, gs in gold_by_img.items()}

    hits = []
    for p in sorted(preds, key=lambda b: b.score, reverse=True):
        best_i, best_iou = -1, iou_threshold
        for i, g in enumerate(gold_by_img.get(p.image_id, [])):
            if not claimed[p.image_id][i]:
                v = iou(p.bbox, g.bbox)
                if v >= best_iou:
                    best_i, best_iou = i, v
        if best_i >= 0:
            claimed[p.image_id][best_i] = True
        hits.append(best_i >= 0)

    recalls, precisions, tp = [], [], 0
    for n, hit in enumerate(hits, 1):
        tp += hit
        recalls.append(tp / len(golds))
        precisions.append(tp / n)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap, prev_r = 0.0, 0.0
    for r, p in zip(recalls, precisions):
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def map50(preds: Iterable[Box], golds: Iterable[Box], iou_threshold: float = 0.5) -> dict[str, Optional[float]]:
    """{class: AP} for every class with gold boxes, plus ALL = mean over those classes."""
    preds, golds = list(preds), list(golds)
    out: dict[str, Optional[float]] = {}
    for cls in sorted({g.category for g in golds}):
        out[cls] = average_precision([p for p in preds if p.category == cls],
                                     [g for g in golds if g.category == cls], iou_threshold)
    vals = [v for v in out.values() if v is not None]
    out[ALL] = sum(vals) / len(vals) if vals else None
    return out
