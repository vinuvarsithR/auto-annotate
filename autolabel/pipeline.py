"""
pipeline.py - run a detector over a list of images (docs/CONTRACT.md section 3).

The backend worker calls run() and gets results back ONE IMAGE AT A TIME,
so it can save boxes and update job progress as it goes.
This file never touches the database and never imports torch itself.
"""

from __future__ import annotations

from typing import Callable, Iterable, Iterator

from PIL import Image

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
