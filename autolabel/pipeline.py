"""
pipeline.py - run a detector over a list of images (docs/CONTRACT.md section 3).

The backend worker calls run() and gets results back ONE IMAGE AT A TIME,
so it can save boxes and update job progress as it goes.
This file never touches the database and never imports torch itself.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

from PIL import Image

from autolabel.evaluate import Box, iou
from autolabel.models import get_detector, list_detectors
from autolabel.models.base import ClassSpec, Prediction


def run(images: Iterable[tuple[str, str]],            # (image_id, path on disk)
        classes: list[ClassSpec],
        detector: str = "pipeline",
        options: dict | None = None,
        on_progress: Callable[[int, int], None] | None = None,   # (done, total)
        ) -> Iterator[tuple[str, list[Prediction]]]:
    """Yield (image_id, predictions) for each image, in input order."""
    if detector not in list_detectors():
        raise ValueError(
            f"detector {detector!r} is not available yet; choose one of {list_detectors()}"
        )
    det = get_detector(detector, **(options or {}))
    items = list(images)
    total = len(items)
    for done, (image_id, path) in enumerate(items, start=1):
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        preds = det.detect(rgb, image_id, classes)
        if on_progress is not None:
            on_progress(done, total)
        yield image_id, preds


def drop_overlapping(predictions: list[Prediction],
                     existing: list[Box],              # the image's non-rejected boxes, category = class name
                     iou_threshold: float = 0.5,
                     ) -> list[Prediction]:
    """Drop each prediction that overlaps an existing box of the same class at IoU >= iou_threshold.

    The caller passes only non-rejected boxes, so a rejected box never blocks a new one.
    Any source counts (human, import, reviewed model box). Order of the kept ones is unchanged.
    """
    return [
        p for p in predictions
        if not any(b.category == p.category and iou(p.bbox, b.bbox) >= iou_threshold
                   for b in existing)
    ]