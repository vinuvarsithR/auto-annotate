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

Examples mode: the usual boxes, for the classes that have examples. Example images are
opened, so an unreadable one fails the job exactly as with a real detector.
"""

from __future__ import annotations

from PIL import Image

from autolabel.models.base import ClassSpec, Example, ExampleRun, Prediction

# (relative x, y, w, h, score) for box 1 and box 2
_LAYOUT = [
    (0.1, 0.1, 0.3, 0.3, 0.9),
    (0.5, 0.5, 0.3, 0.3, 0.3),
]


class FakeExampleRun:
    def __init__(self, detector: "FakeDetector", classes: list[ClassSpec], used: dict[str, int]) -> None:
        self.detector = detector
        self.classes = classes
        self.used = used
        self.unusable: list[str] = []

    def detect(self, image: Image.Image, image_id: str) -> list[Prediction]:
        return self.detector.detect(image, image_id, self.classes)


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

    def prepare_examples(self, classes: list[ClassSpec],
                         examples: dict[str, list[Example]]) -> ExampleRun:
        ran = [c for c in classes if examples.get(c.name)]
        for c in ran:
            for ex in examples[c.name]:
                with Image.open(ex.path) as im:
                    im.load()
        return FakeExampleRun(self, ran, {c.name: len(examples[c.name]) for c in ran})
