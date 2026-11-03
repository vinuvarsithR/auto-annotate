"""plan_import: read an unpacked zip or a saved folder upload (CONTRACT #22).

Format detection order (first match wins): coco, yolo, images. Never touches the database;
opens images only to read their size.
"""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image

from autolabel.importers.base import ImportItem, ImportPlan, is_image_name, skip
from autolabel.importers.coco import coco_boxes, find_coco
from autolabel.importers.yolo import find_yolo, yolo_boxes


def list_files(root: Path) -> list[str]:
    """Regular files under root as '/'-separated relative paths, sorted (path order)."""
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = Path(dirpath) / fn
            if full.is_file() and not full.is_symlink():
                out.append(full.relative_to(root).as_posix())
    return sorted(out)


def image_size(path: Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as im:
            w, h = im.size
    except Exception:  # noqa: BLE001 - any failure means "not a usable image"
        return None
    return (w, h) if w > 0 and h > 0 else None


def plan_import(root: Path) -> ImportPlan:
    root = Path(root)
    files = list_files(root)
    image_files = [f for f in files if is_image_name(f)]
    skipped: list[dict] = []
    sizes: dict[str, tuple[int, int]] = {}
    for rel in image_files:
        size = image_size(root / rel)
        if size is None:
            skipped.append(skip(rel, "cannot open image"))
        else:
            sizes[rel] = size

    boxes: dict[str, list[dict]] = {}
    class_names: list[str] = []
    label_files: set[str] = set()
    coco_files = find_coco(root, files)
    if coco_files:
        fmt = "coco"
        boxes, class_names, more = coco_boxes(coco_files, image_files, sizes)
        skipped += more
        label_files = {rel for rel, _ in coco_files}
    elif (cfg := find_yolo(root, files)) is not None:
        fmt = "yolo"
        boxes, class_names, more, label_files = yolo_boxes(root, cfg, image_files, sizes)
        skipped += more
    else:
        fmt = "images"

    reported = {s["file_name"] for s in skipped}
    for rel in files:
        if not is_image_name(rel) and rel not in label_files and rel not in reported:
            skipped.append(skip(rel, "not an image"))
    skipped.sort(key=lambda s: s["file_name"])  # stable: line errors keep their order

    items = [ImportItem(rel, boxes.get(rel, [])) for rel in image_files if rel in sizes]
    return ImportPlan(format=fmt, items=items, class_names=class_names, skipped=skipped)
