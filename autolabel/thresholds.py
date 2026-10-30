"""
thresholds.py - pick a confidence threshold per class from a sweep.

Plain-English summary
---------------------
* A higher threshold drops more boxes: less junk, but more misses.
* For each class we want the HIGHEST threshold that still keeps recall at or
  above a target (e.g. 0.90). That gives the least junk for the recall we need.
* If no threshold reaches the target, we fall back to the one with the best
  recall (ties -> the higher threshold) and mark the class 'met = False'.
* A class with no gold boxes has no recall, so no threshold can be chosen.

Input is the flat rows from evaluate.sweep(). Pure logic, no GPU.
"""

from __future__ import annotations

from typing import Iterable

from autolabel.evaluate import ALL


def select_thresholds(rows: Iterable[dict], target_recall: float) -> dict[str, dict]:
    """
    Returns {class_name: {threshold, precision, recall, n_gold, met}}.
    threshold/precision/recall are None for a class with no gold boxes.
    The ALL (overall) row is skipped: it isn't a class.
    """
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        if r["class"] != ALL:
            by_class.setdefault(r["class"], []).append(r)

    chosen: dict[str, dict] = {}
    for cls, class_rows in sorted(by_class.items()):
        scored = [r for r in class_rows if r["recall"] is not None]
        if not scored:
            chosen[cls] = {"threshold": None, "precision": None, "recall": None, "n_gold": 0, "met": False}
            continue

        meeting = [r for r in scored if r["recall"] >= target_recall]
        if meeting:
            best = max(meeting, key=lambda r: r["threshold"])
        else:
            best = max(scored, key=lambda r: (r["recall"], r["threshold"]))
        chosen[cls] = {
            "threshold": best["threshold"],
            "precision": best["precision"],
            "recall": best["recall"],
            "n_gold": best["n_gold"],
            "met": bool(meeting),
        }
    return chosen


def apply_thresholds(preds, chosen: dict[str, dict]) -> list:
    """
    Keep each predicted Box only if its score reaches its class's chosen threshold.
    Classes with no chosen threshold (no gold boxes where they were picked) keep nothing.
    """
    out = []
    for p in preds:
        c = chosen.get(p.category)
        if c is not None and c["threshold"] is not None and p.score >= c["threshold"]:
            out.append(p)
    return out
