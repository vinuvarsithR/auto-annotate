"""Shared types and helpers for bulk import (CONTRACT #22)."""

from __future__ import annotations

from dataclasses import dataclass

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class ImportRejected(ValueError):
    """The upload is refused before a job exists. The message is safe to show in a 400."""


@dataclass(frozen=True)
class ImportItem:
    rel_path: str            # path under the root, '/' separators
    boxes: list[dict]        # [{"class_name": "biax", "bbox": [x, y, w, h]}], px, clipped


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


def skip(file_name: str, reason: str) -> dict:
    return {"file_name": file_name, "reason": reason}
