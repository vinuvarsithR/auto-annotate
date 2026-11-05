"""Shared types and helpers for bulk import (CONTRACT #22)."""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
MIN_POLYGON_AREA = 1e-12   # normalized; float noise on collinear points must count as zero


class ImportRejected(ValueError):
    """The upload is refused before a job exists. The message is safe to show in a 400."""


@dataclass(frozen=True)
class ImportItem:
    rel_path: str            # path under the root, '/' separators
    boxes: list[dict]        # [{"class_name": "biax", "bbox": [x, y, w, h], "polygon": [[x, y], ...] | None}]
                             # bbox px, clipped; polygon normalized, clamped; bbox = polygon's bbox when set


@dataclass
class ImportPlan:
    format: str              # "coco" | "yolo" | "images"
    items: list[ImportItem]  # one per image file, in path order
    class_names: list[str]   # names with >= 1 box, in file order
    skipped: list[dict]      # [{"file_name", "reason"}]


def is_image_name(name: str) -> bool:
    dot = name.rfind(".")
    return dot >= 0 and name[dot:].lower() in IMAGE_EXTS


def clip_box(x: float, y: float, w: float, h: float, width: int, height: int) -> list[float] | None:
    """Clip [x, y, w, h] px to the image. None when no area is left."""
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(width), x + w), min(float(height), y + h)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def polygon_area(points: list[list[float]]) -> float:
    """Shoelace area, in the units of the points (normalized or pixels)."""
    pairs = zip(points, points[1:] + points[:1])
    return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in pairs)) / 2


def clamp_polygon(points: list[list[float]]) -> list[list[float]] | None:
    """Clamp normalized points to 0..1. None when no area is left."""
    out = [[min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)] for x, y in points]
    return out if polygon_area(out) > MIN_POLYGON_AREA else None


def polygon_bbox(points: list[list[float]], width: int, height: int) -> list[float]:
    """[x, y, w, h] px of normalized points."""
    xs, ys = [p[0] * width for p in points], [p[1] * height for p in points]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def skip(file_name: str, reason: str) -> dict:
    return {"file_name": file_name, "reason": reason}
