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
  `query_embedding` of its examples is one query. Raw sigmoids sit near 0.9995 on the crane
  data, so the score is calibrated per class: sigmoid(logit - class_cutoff), where class_cutoff
  is the median logit of the class's own examples against the prototype. 0.5 means "as good
  as a typical example". After NMS only the TOP_PER_CLASS best boxes per class per image stay.
* Weights download once into the Hugging Face cache (~/.cache/huggingface,
  or HF_HOME); get_detector() keeps one loaded instance per process.
"""

from __future__ import annotations

from collections import Counter
from statistics import median

from PIL import Image

from autolabel.models.base import ClassSpec, Example, ExampleRun, Prediction
from autolabel.owlv2 import Detection, best_box_index, xywh_to_rel_xyxy

DEFAULT_MIN_SCORE = 0.05
TOP_PER_CLASS = 5       # examples mode only, fixed by CONTRACT


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


def top_per_class(preds: list[Prediction], limit: int = TOP_PER_CLASS) -> list[Prediction]:
    """Keep the first `limit` boxes of each class. preds must be best-first."""
    out, seen = [], Counter()
    for p in preds:
        seen[p.category] += 1
        if seen[p.category] <= limit:
            out.append(p)
    return out


class Owlv2ExampleRun:
    def __init__(self, backend, classes: list[ClassSpec], prototypes: list, cutoffs: list[float],
                 floors: list[float], used: dict[str, int], nms_iou: float = 0.5) -> None:
        self.backend = backend
        self.classes = classes
        self.prototypes = prototypes
        self.cutoffs = cutoffs
        self.floors = floors
        self.used = used
        self.nms_iou = nms_iou
        self.unusable: list[str] = []

    def detect(self, image: Image.Image, image_id: str) -> list[Prediction]:
        if not self.classes:
            return []
        import torch
        from torchvision.ops import batched_nms

        logits, boxes = self.backend.image_guided_logits(image, self.prototypes)
        margin = logits - torch.tensor(self.cutoffs, dtype=logits.dtype)
        scores = torch.sigmoid(margin)
        patch, label = (scores >= torch.tensor(self.floors, dtype=scores.dtype)).nonzero(as_tuple=True)
        if len(patch) == 0:
            return []

        width, height = image.size
        size = float(max(width, height))
        cx, cy, bw, bh = boxes[patch].unbind(-1)
        xyxy = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], -1) * size
        # margin, not score: far above the cutoff the sigmoid rounds to exactly 1.0 and order is lost.
        kept = batched_nms(xyxy, margin[patch, label], label, self.nms_iou)   # best first

        dets = []
        for i in kept.tolist():
            x1, y1, x2, y2 = xyxy[i].tolist()
            # Clip to the real image: the padded area can produce boxes past the edge.
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            c = int(label[i])
            dets.append(Detection(c, float(scores[patch[i], c]), (x1, y1, x2 - x1, y2 - y1)))
        return top_per_class(to_predictions(dets, self.classes, self.floors, image_id, "owlv2_image"))


class Owlv2TextDetector:
    name = "owlv2"

    def __init__(self, model_id: str | None = None, device: str = "auto",
                 nms_iou: float = 0.5, min_score: float = DEFAULT_MIN_SCORE,
                 backend=None) -> None:
        """backend: anything with detect(image, prompts, threshold) -> list[owlv2.Detection].
        Left as None, the real OWLv2 model is loaded (torch imported here, not at module level)."""
        self.min_score = min_score
        self.nms_iou = nms_iou
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
        cutoffs = self._class_cutoffs(ran, prototypes, examples)
        return Owlv2ExampleRun(self.backend, ran, prototypes, cutoffs, floors, used, self.nms_iou)

    def _class_cutoffs(self, classes: list[ClassSpec], prototypes: list,
                       examples: dict[str, list[Example]]) -> list[float]:
        """Per class, the median logit against its prototype of the patches query_embedding picked.
        One model pass per example image, whatever the number of examples in it."""
        by_image: dict[str, list[tuple[int, tuple]]] = {}
        for c, cls in enumerate(classes):
            for ex in examples[cls.name]:
                by_image.setdefault(ex.path, []).append((c, ex.bbox))

        found: list[list[float]] = [[] for _ in classes]
        for path, items in by_image.items():
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            logits, boxes = self.backend.image_guided_logits(rgb, prototypes)
            for c, bbox in items:
                i, _overlap = best_box_index(boxes, xywh_to_rel_xyxy(list(bbox), max(rgb.size)))
                found[c].append(float(logits[i, c]))
        return [median(values) for values in found]
