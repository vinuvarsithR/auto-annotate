"""
sam.py - OWLv2 + SAM 2 steps 3 and 4 (docs/CONTRACT.md "OWLv2 + SAM 2 polygons").

Plain-English summary
---------------------
* A Segmenter turns boxes into masks. The real one is SAM 2.1 tiny (ultralytics, CPU); tests use
  FakeSegmenter (an ellipse inside each box), so no weights are needed.
* clean_mask (step 3): SAM gives 3 candidate masks per box. Keep the best-scoring one that stays
  inside the box, cut thin strands (cables) with a 5 px morphological opening, clear pixels outside the box.
* largest_polygon (step 4): mask -> the outline of its largest part, as a 0..1 polygon.
* add_polygons: steps 3 + 4 for all boxes of one image. A box whose mask gives no usable polygon
  is kept as a plain box (polygon None).

Weights: AUTOLABEL_SAM_WEIGHTS, else models/sam2.1_t.pt under the repo root. Never downloads.
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from autolabel.models.base import Prediction
from autolabel.models.yoloe import SIMPLIFY_TOLERANCE, polygon_bbox

ENV_WEIGHTS = "AUTOLABEL_SAM_WEIGHTS"
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "sam2.1_t.pt"
INSIDE_FRAC = 0.9      # share of a mask's pixels that must be inside the grown box
INSIDE_PAD = 5         # px the box is grown by on each side for that check
OPEN_PX = 5            # opening kernel size (ellipse)


class Segmenter(Protocol):
    def segment(self, image: Image.Image, boxes: list[tuple]) -> list[tuple[np.ndarray, np.ndarray]]: ...
    # boxes (x, y, w, h) px; per box: (masks bool k x H x W at image size, scores float k), k >= 1


class FakeSegmenter:
    """Tests: per box one mask (k = 1), the filled ellipse inscribed in the box, score 0.9."""

    def segment(self, image: Image.Image, boxes: list[tuple]) -> list[tuple[np.ndarray, np.ndarray]]:
        import cv2

        width, height = image.size
        out = []
        for x, y, w, h in boxes:
            m = np.zeros((height, width), np.uint8)
            center = (int(round(x + w / 2)), int(round(y + h / 2)))
            axes = (max(int(round(w / 2)), 1), max(int(round(h / 2)), 1))
            cv2.ellipse(m, center, axes, 0, 0, 360, 1, thickness=-1)
            out.append((m[None].astype(bool), np.array([0.9], np.float32)))
        return out


class Sam2Segmenter:
    """SAM 2.1 tiny through ultralytics, CPU. One image-encoder pass per image, all boxes in one call,
    multimask_output=True (call path verified in spike v2 steps.py, ultralytics 8.4.199)."""

    def __init__(self, weights: str):
        from autolabel.models.yolo import disable_autoinstall

        disable_autoinstall()
        from ultralytics import SAM

        self.model = SAM(weights)
        # ultralytics builds its predictor on the first call; a tiny call sets it up once.
        self.model(np.zeros((64, 64, 3), np.uint8), bboxes=[[1, 1, 30, 30]], device="cpu", verbose=False)
        self.predictor = self.model.predictor

    def segment(self, image: Image.Image, boxes: list[tuple]) -> list[tuple[np.ndarray, np.ndarray]]:
        if not boxes:
            return []
        width, height = image.size
        bgr = np.ascontiguousarray(np.asarray(image.convert("RGB"))[:, :, ::-1])
        p = self.predictor
        p.set_image(bgr)
        xyxy = np.array([[x, y, x + w, y + h] for x, y, w, h in boxes], np.float32)
        masks, rows = p.inference_features(p.features, src_shape=(height, width), bboxes=xyxy,
                                           multimask_output=True)
        masks = masks.cpu().numpy().astype(bool)
        scores = rows[:, 4].cpu().numpy().astype(np.float32)
        k = len(masks) // len(boxes)
        return [(masks[i * k:(i + 1) * k], scores[i * k:(i + 1) * k]) for i in range(len(boxes))]


_CACHE: dict[str, Sam2Segmenter] = {}


def resolve_weights() -> str:
    """AUTOLABEL_SAM_WEIGHTS > models/sam2.1_t.pt. Checks the file; never downloads."""
    path = os.environ.get(ENV_WEIGHTS, "").strip() or str(DEFAULT_WEIGHTS)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"SAM weights not found: {path} (set {ENV_WEIGHTS})")
    return path


def get_segmenter() -> Segmenter:
    """SAM 2.1 tiny, loaded on first use and cached per process (per weights path)."""
    path = resolve_weights()
    if path not in _CACHE:
        _CACHE[path] = Sam2Segmenter(path)
    return _CACHE[path]


def _inside_fraction(mask: np.ndarray, bbox: tuple, pad: float) -> float:
    total = int(mask.sum())
    if total == 0:
        return 0.0
    x, y, w, h = bbox
    r0, r1 = max(0, math.floor(y - pad)), math.ceil(y + h + pad)
    c0, c1 = max(0, math.floor(x - pad)), math.ceil(x + w + pad)
    return float(mask[r0:r1 + 1, c0:c1 + 1].sum()) / total


def clean_mask(masks, scores, bbox: tuple, open_px: int = OPEN_PX) -> np.ndarray:
    """Step 3. masks: k x H x W, scores: k, bbox (x, y, w, h) px. Returns one bool H x W mask.

    Pick: among non-empty masks with >= 90% of pixels inside the box grown by 5 px, the highest
    score (first on a tie); none qualifies -> the highest inside fraction. Opening with an
    open_px x open_px ellipse (kept unopened if that empties it). Clip to the box
    (floor(x0)..ceil(x1), floor(y0)..ceil(y1), inclusive)."""
    import cv2

    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(scores, dtype=float)
    inside = [_inside_fraction(m, bbox, INSIDE_PAD) for m in masks]
    ok = [i for i, m in enumerate(masks) if m.any() and inside[i] >= INSIDE_FRAC]
    if ok:
        pick = max(ok, key=lambda i: (scores[i], -i))
    else:
        pick = max(range(len(masks)), key=lambda i: (inside[i], -i))
    mask = masks[pick].copy()
    if open_px and mask.any():
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px))
        opened = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
        if opened.any():
            mask = opened
    x, y, w, h = bbox
    keep = np.zeros_like(mask)
    keep[max(0, math.floor(y)):math.ceil(y + h) + 1, max(0, math.floor(x)):math.ceil(x + w) + 1] = True
    return mask & keep


def largest_polygon(mask: np.ndarray, width: int, height: int) -> list | None:
    """Step 4. The largest part's outer ring (holes dropped), simplified as yoloe, 0..1, 6 decimals.
    None (= keep the box, polygon None) for an empty mask, < 3 points or zero area."""
    import cv2

    binary = (np.asarray(mask) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)  # max() keeps the first on a tie
    points = cv2.approxPolyDP(largest, SIMPLIFY_TOLERANCE * max(width, height), True)
    if len(points) < 3 or cv2.contourArea(points) <= 0:
        return None
    return [[round(float(px) / width, 6), round(float(py) / height, 6)] for px, py in points[:, 0]]


def add_polygons(image: Image.Image, preds: list[Prediction], segmenter: Segmenter) -> list[Prediction]:
    """Steps 3 + 4 for one image. With a polygon: bbox = the polygon's box, mask None.
    Without one: the box is kept unchanged with polygon None and mask None."""
    if not preds:
        return []
    width, height = image.size
    results = segmenter.segment(image, [tuple(p.bbox) for p in preds])
    out = []
    for p, (masks, scores) in zip(preds, results):
        polygon = largest_polygon(clean_mask(masks, scores, tuple(p.bbox)), width, height)
        if polygon is None:
            out.append(replace(p, polygon=None, mask=None))
        else:
            out.append(replace(p, bbox=polygon_bbox(polygon, width, height), polygon=polygon, mask=None))
    return out
