"""
owlv2.py - the real OWLv2 text-prompt detector, behind the same interface as "fake".

Plain-English summary
---------------------
* One text prompt per class: the class's `description`, or its `name` when the
  description is empty. (ClassSpec.description IS the per-class prompt override.)
* Score floor per class: the class's `min_score`, or DEFAULT_MIN_SCORE (0.05).
  0.05 is the lowest threshold of the Phase 1 sweep (CLAUDE.md "Run 1" and
  "Step A"; also the min_score used in docs/CONTRACT.md examples). Phase 1 found
  0.20 far too high for the crane data and the project favours recall.
* The heavy work is done by autolabel/owlv2.py (read-only, Phase 1 code). This
  file only turns its output into Prediction objects.
* Weights download once into the Hugging Face cache (~/.cache/huggingface,
  or HF_HOME); get_detector() keeps one loaded instance per process.
"""

from __future__ import annotations

from PIL import Image

from autolabel.models.base import ClassSpec, Prediction

DEFAULT_MIN_SCORE = 0.05


def prompt_for(cls: ClassSpec) -> str:
    """The text prompt for a class: its description, else its name."""
    return (cls.description or "").strip() or cls.name


def floor_for(cls: ClassSpec, default: float) -> float:
    return default if cls.min_score is None else cls.min_score


class Owlv2TextDetector:
    name = "owlv2"

    def __init__(self, model_id: str | None = None, device: str = "auto",
                 nms_iou: float = 0.5, min_score: float = DEFAULT_MIN_SCORE,
                 backend=None) -> None:
        """backend: anything with detect(image, prompts, threshold) -> list[owlv2.Detection].
        Left as None, the real OWLv2 model is loaded (torch imported here, not at module level)."""
        self.min_score = min_score
        if backend is None:
            from autolabel.owlv2 import DEFAULT_MODEL, Owlv2Detector
            backend = Owlv2Detector(model_id or DEFAULT_MODEL, device=device, nms_iou=nms_iou)
        self.backend = backend

    def detect(self, image: Image.Image, image_id: str,
               classes: list[ClassSpec]) -> list[Prediction]:
        if not classes:
            return []
        prompts = [prompt_for(c) for c in classes]
        floors = [floor_for(c, self.min_score) for c in classes]
        # Ask for everything above the lowest floor, then apply each class's own floor.
        dets = self.backend.detect(image, prompts, threshold=min(floors))
        out: list[Prediction] = []
        for d in dets:
            if not 0 <= d.label < len(classes):
                continue
            score = min(1.0, max(0.0, float(d.score)))
            if score < floors[d.label]:
                continue
            out.append(Prediction(
                image_id=image_id,
                category=classes[d.label].name,
                bbox=tuple(float(v) for v in d.bbox),
                score=score,
                source=self.name,
                mask=None,
            ))
        return out
