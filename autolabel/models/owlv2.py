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
* Examples mode (source "owlv2_image"): per class, the mean of the normalized
  `query_embedding` of its examples is one query for `detect_image_guided`. The score
  is its sigmoid, 0..1 like every mode, but on a very different scale from text
  scores (best-F1 thresholds near 0.9995 on the crane data, VERIFIED in the spike).
* Weights download once into the Hugging Face cache (~/.cache/huggingface,
  or HF_HOME); get_detector() keeps one loaded instance per process.
"""

from __future__ import annotations

from PIL import Image

from autolabel.models.base import ClassSpec, Example, ExampleRun, Prediction

DEFAULT_MIN_SCORE = 0.05


def prompt_for(cls: ClassSpec) -> str:
    """The text prompt for a class: its description, else its name."""
    return (cls.description or "").strip() or cls.name


def floor_for(cls: ClassSpec, default: float) -> float:
    return default if cls.min_score is None else cls.min_score


def to_predictions(dets, classes: list[ClassSpec], floors: list[float],
                   image_id: str, source: str) -> list[Prediction]:
    """backend Detections -> Predictions, each class's own floor applied."""
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
            source=source,
            mask=None,
        ))
    return out


class Owlv2ExampleRun:
    def __init__(self, backend, classes: list[ClassSpec], prototypes: list,
                 floors: list[float], used: dict[str, int]) -> None:
        self.backend = backend
        self.classes = classes
        self.prototypes = prototypes
        self.floors = floors
        self.used = used
        self.unusable: list[str] = []

    def detect(self, image: Image.Image, image_id: str) -> list[Prediction]:
        if not self.classes:
            return []
        queries = [[p] for p in self.prototypes]
        dets = self.backend.detect_image_guided(image, queries, threshold=min(self.floors))
        return to_predictions(dets, self.classes, self.floors, image_id, "owlv2_image")


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
        return to_predictions(dets, classes, floors, image_id, self.name)

    def prepare_examples(self, classes: list[ClassSpec],
                         examples: dict[str, list[Example]]) -> ExampleRun:
        import torch
        from torch.nn.functional import normalize

        ran, prototypes, used = [], [], {}
        for c in classes:
            items = examples.get(c.name) or []
            if not items:
                continue
            embeddings = []
            for ex in items:
                with Image.open(ex.path) as im:
                    rgb = im.convert("RGB")
                embedding, _overlap = self.backend.query_embedding(rgb, list(ex.bbox))
                embeddings.append(normalize(embedding, dim=-1))
            ran.append(c)
            prototypes.append(normalize(torch.stack(embeddings).mean(0), dim=-1))
            used[c.name] = len(items)
        floors = [floor_for(c, self.min_score) for c in ran]
        return Owlv2ExampleRun(self.backend, ran, prototypes, floors, used)
