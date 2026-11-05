"""
yoloe.py - YOLOE visual prompts: find more objects like the project's own boxes, no training.

Plain-English summary
---------------------
* Examples mode only (CONTRACT #9), source "yoloe_examples". There is no text mode: text prompts
  need extra packages, a 254 MB encoder, and scored worse in the spike.
* prepare_examples() loads a FRESH model for each job. A class gets one embedding per image: all
  its boxes in that image form one prompt. Its prototype is the mean of those embeddings,
  L2-normalized, given to set_classes. Embedding each box alone scored much worse (spike frames:
  person F1 0.62 -> 0.52, biax 0.27 -> 0.01; VERIFIED, small sample).
  The run object owns that model, so the instance from get_detector keeps no job state.
* Weights: the `weights` option, else AUTOLABEL_YOLOE_WEIGHTS, else models/yoloe-26s-seg.pt under
  the repo root. Always a `-seg` file, never `-seg-pf`. A missing file fails the job with a
  one-line error; nothing is ever downloaded at job time.
* Image size 640, for embedding and for detection.
* An example whose shorter side is under 8 px at 640 is not used: prompts are rasterized on a
  1/8 grid, and an empty prompt silently turns into a whole-image embedding.
* Masks become polygons with mask_to_polygon(), a pure function. Without a polygon a box keeps
  the model's own bbox; with one, the bbox is the polygon's.
* The score is `boxes.conf` as returned. predict gets conf = the lowest class floor, because the
  ultralytics default would hide low scores; each class's own floor is applied afterwards.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from autolabel.models.base import ClassSpec, Example, ExampleRun, Prediction
from autolabel.models.yolo import _as_list

ENV_WEIGHTS = "AUTOLABEL_YOLOE_WEIGHTS"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "yoloe-26s-seg.pt"
DEFAULT_MIN_SCORE = 0.05
SOURCE = "yoloe_examples"
IMGSZ = 640
MIN_EXAMPLE_SIDE = 8.0     # px, after scaling the image to IMGSZ
MIN_LARGEST_PART = 0.9     # share of the summed part area the largest part must hold
SIMPLIFY_TOLERANCE = 0.001 # times the image's longest side


def resolve_weights(option: str | None = None) -> str:
    """option > AUTOLABEL_YOLOE_WEIGHTS > models/yoloe-26s-seg.pt. Checks the file; never downloads."""
    path = option or os.environ.get(ENV_WEIGHTS, "").strip() or str(DEFAULT_WEIGHTS)
    if not Path(path).stem.endswith("-seg"):
        raise ValueError(f"yoloe weights must be a -seg file: {path}")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"yoloe weights not found: {path}")
    return path


def mask_to_polygon(mask) -> list[list[float]] | None:
    """Mask (height x width, foreground > 0.5) -> [[x, y], ...] in 0..1, or None (CONTRACT #9).

    One part: use it. Several parts: use the largest only if it holds >= 90% of the summed area,
    because long beams often split and the largest piece alone would shrink the box.
    None also for fewer than 3 points or zero area. Holes are dropped.
    """
    import cv2

    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    height, width = binary.shape
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    areas = [cv2.contourArea(c) for c in contours]
    largest = int(np.argmax(areas))
    if len(contours) > 1 and areas[largest] < MIN_LARGEST_PART * sum(areas):
        return None
    tolerance = SIMPLIFY_TOLERANCE * max(width, height)
    points = cv2.approxPolyDP(contours[largest], tolerance, True)
    if len(points) < 3 or cv2.contourArea(points) <= 0:
        return None
    return [[round(float(x) / width, 6), round(float(y) / height, 6)] for x, y in points[:, 0]]


def polygon_bbox(polygon: list, width: int, height: int) -> tuple[float, float, float, float]:
    """(x, y, w, h) px: the polygon's bounding box."""
    xs = [p[0] * width for p in polygon]
    ys = [p[1] * height for p in polygon]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)


def usable(bbox: tuple, image_size: tuple[int, int]) -> bool:
    """True when the box's shorter side is at least 8 px once the image is scaled to 640."""
    scale = IMGSZ / max(image_size)
    return min(bbox[2], bbox[3]) * scale >= MIN_EXAMPLE_SIDE


def disable_autoinstall() -> None:
    """Never pip-install during a job.

    ultralytics patches PIL.Image.open: after that, any file Pillow cannot identify runs
    `pip install pi-heif` in the live environment (VERIFIED 2026-11-05, 8.4.199). The env variable
    is read once at import, so also set the flag, in case ultralytics was imported earlier.
    """
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics.utils import checks

    checks.AUTOINSTALL = False


class UltralyticsYoloe:
    """The few YOLOE calls we use, in one place, so tests can swap the model for a fake."""

    def __init__(self, weights: str, device: str) -> None:
        disable_autoinstall()
        from ultralytics import YOLOE

        self.model = YOLOE(weights)
        self.device = device

    def embed(self, image: Image.Image, boxes: np.ndarray, labels: np.ndarray):
        """One embedding per label, from one pass over the image -> [labels, dim].

        boxes: [n, 4] x1, y1, x2, y2 px. labels: [n], 0..k-1; boxes sharing a label form one prompt.
        Calls the predictor directly: unlike model.predict(refer_image=...) this runs no detection
        and works many times on one model. Same embeddings (cosine 1.0, checked 2026-11-05).
        """
        from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

        predictor = YOLOEVPSegPredictor(overrides=dict(
            task="segment", mode="predict", save=False, batch=1, imgsz=IMGSZ,
            device=self.device, verbose=False))
        predictor.set_prompts(dict(bboxes=boxes, cls=labels))
        predictor.setup_model(model=self.model.model, verbose=False)
        return predictor.get_vpe(image)[0].detach().clone()

    def set_classes(self, names: list[str], embeddings) -> None:
        self.model.set_classes(names, embeddings)

    def predict(self, image: Image.Image, conf: float):
        # Each class is scored on its own: YOLOE's default class-agnostic NMS lets a neighbouring
        # class hide a box. Same setting as the spike.
        return self.model.predict(image, imgsz=IMGSZ, conf=conf, retina_masks=True,
                                  agnostic_nms=False, device=self.device, verbose=False)[0]


class YoloeExampleRun:
    def __init__(self, model, classes: list[ClassSpec], used: dict[str, int],
                 unusable_names: list[str], min_score: float) -> None:
        self.model = model
        self.classes = classes
        self.used = used
        self.unusable = unusable_names
        self.floors = [min_score if c.min_score is None else c.min_score for c in classes]

    def detect(self, image: Image.Image, image_id: str) -> list[Prediction]:
        if not self.classes:
            return []
        result = self.model.predict(image, conf=min(self.floors))
        boxes = result.boxes
        masks = result.masks.data if result.masks is not None else None
        width, height = image.size
        out: list[Prediction] = []
        rows = zip(_as_list(boxes.xyxy), _as_list(boxes.conf), _as_list(boxes.cls))
        for i, ((x1, y1, x2, y2), score, label) in enumerate(rows):
            label = int(label)
            if not 0 <= label < len(self.classes):
                continue
            score = min(1.0, max(0.0, float(score)))
            if score < self.floors[label]:
                continue
            polygon = None
            if masks is not None:
                mask = masks[i]
                polygon = mask_to_polygon(mask.cpu().numpy() if hasattr(mask, "cpu") else mask)
            bbox = (polygon_bbox(polygon, width, height) if polygon
                    else (float(x1), float(y1), float(x2 - x1), float(y2 - y1)))
            out.append(Prediction(
                image_id=image_id,
                category=self.classes[label].name,
                bbox=bbox,
                score=score,
                source=SOURCE,
                mask=None,
                polygon=polygon,
            ))
        return out


class YoloeDetector:
    name = "yoloe"

    def __init__(self, weights: str | None = None, device: str = "cpu",
                 min_score: float = DEFAULT_MIN_SCORE, loader=None) -> None:
        """loader(weights, device) -> a model like UltralyticsYoloe. Tests pass a fake."""
        self.weights = weights
        self.device = device
        self.min_score = min_score
        self.loader = loader or UltralyticsYoloe

    def detect(self, image: Image.Image, image_id: str,
               classes: list[ClassSpec]) -> list[Prediction]:
        raise RuntimeError("yoloe has no text mode: use pipeline.prepare_examples")

    def prepare_examples(self, classes: list[ClassSpec],
                         examples: dict[str, list[Example]]) -> ExampleRun:
        import torch
        from torch.nn.functional import normalize

        model = self.loader(resolve_weights(self.weights), self.device)

        # One pass per image, whatever classes its examples belong to.
        by_path: dict[str, list[Example]] = {}
        for c in classes:
            for ex in examples.get(c.name) or []:
                by_path.setdefault(ex.path, []).append(ex)

        embeddings: dict[str, list] = {}   # class name -> one embedding per image
        used: dict[str, int] = {}          # class name -> example boxes
        for path, items in by_path.items():
            with Image.open(path) as im:
                rgb = im.convert("RGB")
            items = [ex for ex in items if usable(ex.bbox, rgb.size)]
            if not items:
                continue
            boxes = np.array([(x, y, x + w, y + h) for x, y, w, h in (ex.bbox for ex in items)],
                             dtype=np.float32)
            names = list(dict.fromkeys(ex.category for ex in items))
            labels = np.array([names.index(ex.category) for ex in items])
            for name, vector in zip(names, model.embed(rgb, boxes, labels)):
                embeddings.setdefault(name, []).append(vector)
                used[name] = used.get(name, 0) + sum(ex.category == name for ex in items)

        ran = [c for c in classes if c.name in embeddings]
        rejected = [c.name for c in classes if examples.get(c.name) and c.name not in embeddings]
        if ran:
            prototypes = [normalize(torch.stack(embeddings[c.name]).mean(0), dim=-1) for c in ran]
            model.set_classes([c.name for c in ran], torch.stack(prototypes)[None])
        return YoloeExampleRun(model, ran, used, rejected, self.min_score)
