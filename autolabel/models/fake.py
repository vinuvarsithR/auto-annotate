"""
fake.py - a detector that needs no GPU and no model weights.

It always returns the same 2 boxes per image, so the backend can test the
whole job flow on a CPU-only machine:

| box | class            | score | bbox [x, y, w, h]          |
|-----|------------------|-------|----------------------------|
| 1   | first class      | 0.9   | [0.1W, 0.1H, 0.3W, 0.3H]   |
| 2   | second class     | 0.3   | [0.5W, 0.5H, 0.3W, 0.3H]   |

W, H = image width and height, so boxes always fit inside the image.
Fewer than 2 classes -> fewer boxes. Boxes below a class's min_score are dropped.
"""

from __future__ import annotations

from PIL import Image

from autolabel.models.base import ClassSpec, Prediction

# (relative x, y, w, h, score) for box 1 and box 2
_LAYOUT = [
    (0.1, 0.1, 0.3, 0.3, 0.9),
    (0.5, 0.5, 0.3, 0.3, 0.3),
]


class FakeDetector:
    name = "fake"

    def __init__(self, **options) -> None:
        self.options = options  # accepted and ignored, like a real detector's options

    def detect(self, image: Image.Image, image_id: str,
               classes: list[ClassSpec]) -> list[Prediction]:
        width, height = image.size
        out: list[Prediction] = []
        for cls, (rx, ry, rw, rh, score) in zip(classes, _LAYOUT):
            if cls.min_score is not None and score < cls.min_score:
                continue
            out.append(Prediction(
                image_id=image_id,
                category=cls.name,
                bbox=(rx * width, ry * height, rw * width, rh * height),
                score=score,
                source=self.name,
                mask=None,
            ))
        return out
