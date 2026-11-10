"""
owlv2.py - the real OWLv2 detector, behind the same interface as "fake".

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
* Examples mode (source "owlv2_image", CONTRACT #9 owlv2 rules, the models spike's "rule C"):
  - each example keeps its own embedding: the class embedding of the model's predicted box that
    matches the drawn box (highest objectness among boxes with IoU >= 0.9 x the best IoU);
  - a box's score for a class = raw cosine to the class's examples, the MAX over them (no mean);
  - one threshold per class, from the examples only: pos = median leave-one-out cosine between
    examples, bg = median over the example images of the best score on a box that overlaps no
    example box; threshold = min(pos, (pos + bg) / 2);
  - keep boxes with objectness >= 0.01 and score >= threshold and >= min_score, then one
    class-agnostic NMS at IoU 0.3. No top-N cap.
  - a class needs at least 5 examples, else it is skipped ("needs at least 5 examples").
* Weights download once into the Hugging Face cache (~/.cache/huggingface,
  or HF_HOME); get_detector() keeps one loaded instance per process.
"""

from __future__ import annotations

from statistics import median

from PIL import Image

from autolabel.models.base import ClassSpec, Example, ExampleRun, Prediction
from autolabel.owlv2 import xywh_to_rel_xyxy

DEFAULT_MIN_SCORE = 0.05

# Examples mode, fixed by CONTRACT (not options).
MIN_EXAMPLES = 5
TOO_FEW_REASON = "needs at least 5 examples"
MATCH_FRACTION = 0.9      # example embedding: boxes with IoU >= 0.9 x the best IoU compete on objectness
MIN_OBJECTNESS = 0.01
BG_OVERLAP_IOU = 0.3      # a box with IoU >= 0.3 to an example box is not background
NMS_IOU = 0.3             # class-agnostic


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


# ---- examples mode: small pure functions (tested on their own) ----

def _xyxy(cxcywh):
    import torch

    cx, cy, w, h = torch.as_tensor(cxcywh, dtype=torch.float32).unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def example_box_index(boxes, objectness, target_xyxy) -> int:
    """The predicted box an example uses: among boxes (cx, cy, w, h relative) with IoU >= 0.9 x the
    best IoU to target_xyxy (relative), the one with the highest objectness."""
    import torch
    from torchvision.ops import box_iou

    ious = box_iou(torch.tensor([target_xyxy], dtype=torch.float32), _xyxy(boxes))[0]
    candidates = (ious >= ious.max() * MATCH_FRACTION).nonzero().flatten()
    return int(candidates[torch.as_tensor(objectness)[candidates].argmax()])


def class_scores(embeds, examples: list):
    """embeds [boxes, d] (normalized), examples[c] = [n_c, d] (normalized) -> [boxes, classes]:
    each class's score is the max cosine over its examples."""
    import torch

    return torch.stack([(embeds @ e.T).max(dim=1).values for e in examples], dim=1)


def pos_value(examples) -> float:
    """Median over examples of the best cosine to the class's OTHER examples (leave-one-out)."""
    import torch

    sims = examples @ examples.T
    sims.fill_diagonal_(float("-inf"))
    return float(median(sims.max(dim=1).values.tolist()))


def class_threshold(pos: float, bg: float | None) -> float:
    """Rule C: halfway between pos and bg, never above pos, never below 0. No bg value -> pos."""
    t = pos if bg is None else min(pos, (pos + bg) / 2)
    return max(0.0, t)


def background_score(embeds, boxes, objectness, example_xyxy: list, examples) -> float | None:
    """Best score for one class on a box (objectness >= 0.01) that overlaps none of the class's example
    boxes in this image (IoU < 0.3). None when there is no such box."""
    import torch
    from torchvision.ops import box_iou

    free = torch.as_tensor(objectness) >= MIN_OBJECTNESS
    if example_xyxy:
        ious = box_iou(_xyxy(boxes), torch.tensor(example_xyxy, dtype=torch.float32))
        free &= ious.max(dim=1).values < BG_OVERLAP_IOU
    if not bool(free.any()):
        return None
    return float((embeds[free] @ examples.T).max())


class Owlv2ExampleRun:
    def __init__(self, backend, classes: list[ClassSpec], examples: list, thresholds: list[float],
                 floors: list[float], used: dict[str, int], unusable: list[str] | None = None,
                 skip_reasons: dict[str, str] | None = None) -> None:
        self.backend = backend
        self.classes = classes
        self.examples = examples          # per class: [n, d] normalized example embeddings
        self.thresholds = thresholds      # per class: rule C
        self.floors = floors              # per class: min_score or the default floor
        self.used = used
        self.unusable = list(unusable or [])
        self.skip_reasons = dict(skip_reasons or {})

    def detect(self, image: Image.Image, image_id: str) -> list[Prediction]:
        if not self.classes:
            return []
        import torch
        from torchvision.ops import nms

        embeds, boxes, objectness = self.backend.box_features(image)
        scores = class_scores(embeds, self.examples)                       # [boxes, classes]
        need = torch.tensor([max(t, f) for t, f in zip(self.thresholds, self.floors)], dtype=scores.dtype)
        ok = (scores >= need) & (objectness >= MIN_OBJECTNESS)[:, None]
        box, label = ok.nonzero(as_tuple=True)
        if len(box) == 0:
            return []

        width, height = image.size
        xyxy = _xyxy(boxes[box]) * float(max(width, height))
        score = scores[box, label]
        kept = nms(xyxy, score, NMS_IOU)                                    # class-agnostic, best first

        out = []
        for i in kept.tolist():
            x1, y1, x2, y2 = xyxy[i].tolist()
            # Clip to the real image: the padded area can produce boxes past the edge.
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(width), x2), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Prediction(
                image_id=image_id,
                category=self.classes[int(label[i])].name,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                score=min(1.0, max(0.0, float(score[i]))),
                source="owlv2_image",
                mask=None,
            ))
        return out


class Owlv2TextDetector:
    name = "owlv2"

    def __init__(self, model_id: str | None = None, device: str = "auto",
                 nms_iou: float = 0.5, min_score: float = DEFAULT_MIN_SCORE,
                 backend=None) -> None:
        """backend: anything with detect(image, prompts, threshold) -> list[owlv2.Detection]
        (text mode) and box_features(image) (examples mode).
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

        unusable, reasons, ready = [], {}, []
        for c in classes:
            items = examples.get(c.name) or []
            if not items:
                continue                      # the worker reports these as "no examples"
            if len(items) < MIN_EXAMPLES:
                unusable.append(c.name)
                reasons[c.name] = TOO_FEW_REASON
                continue
            ready.append((c, items))

        # One model pass per example image, shared by all classes and by bg.
        features, sizes = {}, {}
        for _, items in ready:
            for ex in items:
                if ex.path not in features:
                    with Image.open(ex.path) as im:
                        rgb = im.convert("RGB")
                    features[ex.path] = self.backend.box_features(rgb)
                    sizes[ex.path] = float(max(rgb.size))

        ran, embeddings, thresholds, used = [], [], [], {}
        for c, items in ready:
            rows, by_image = [], {}
            for ex in items:
                embeds, boxes, objectness = features[ex.path]
                target = xywh_to_rel_xyxy(list(ex.bbox), sizes[ex.path])
                rows.append(embeds[example_box_index(boxes, objectness, target)])
                by_image.setdefault(ex.path, []).append(target)
            mine = torch.stack(rows)
            bgs = [background_score(*features[path], targets, mine) for path, targets in by_image.items()]
            bgs = [b for b in bgs if b is not None]
            ran.append(c)
            embeddings.append(mine)
            thresholds.append(class_threshold(pos_value(mine), float(median(bgs)) if bgs else None))
            used[c.name] = len(items)
        floors = [floor_for(c, self.min_score) for c in ran]
        return Owlv2ExampleRun(self.backend, ran, embeddings, thresholds, floors, used, unusable, reasons)
