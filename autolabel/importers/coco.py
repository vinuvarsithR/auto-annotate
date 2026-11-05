"""COCO import (CONTRACT #22).

A COCO file is any .json with `images`, `annotations` and `categories` keys. Each file applies
to the images in its own folder and below; images are matched by the base name of
`images[].file_name`. When several files cover an image (nested folders), the deepest file that
lists the image's base name wins.

`segmentation` with one usable polygon part gives a polygon row. Anything else that is not missing
or `[]` (RLE, several parts, unusable) gives a box row, counted in one note per COCO file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path, PurePosixPath

from autolabel.importers.base import clamp_polygon, clip_box, polygon_bbox, skip

COCO_KEYS = ("images", "annotations", "categories")
BOX_ONLY_NOTE = "{} segmentations imported as box only (RLE, multi-part or unusable polygon)"


def _folder(rel: str) -> str:
    parent = str(PurePosixPath(rel).parent)
    return "" if parent == "." else parent


def _in_scope(rel: str, folder: str) -> bool:
    return folder == "" or rel.startswith(folder + "/")


def find_coco(root: Path, files: list[str]) -> list[tuple[str, dict]]:
    """All COCO files in path order, parsed."""
    found = []
    for rel in files:
        if not rel.lower().endswith(".json"):
            continue
        try:
            data = json.loads((root / rel).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, RecursionError):
            continue
        if isinstance(data, dict) and all(isinstance(data.get(k), list) for k in COCO_KEYS):
            found.append((rel, data))
    return found


def _is_id(v) -> bool:
    return isinstance(v, (int, str)) and not isinstance(v, bool)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _bbox(v) -> list[float] | None:
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    if not all(_is_number(n) for n in v):
        return None
    return [float(n) for n in v]


def _polygon(seg, width: int, height: int) -> list[list[float]] | None:
    """One flat px part -> clamped normalized points; None when it is not a usable polygon."""
    if not isinstance(seg, list) or len(seg) != 1 or not isinstance(seg[0], list):
        return None
    flat = seg[0]
    if len(flat) < 6 or len(flat) % 2 or not all(_is_number(v) for v in flat):
        return None
    return clamp_polygon([[flat[i] / width, flat[i + 1] / height] for i in range(0, len(flat), 2)])


def coco_boxes(coco_files: list[tuple[str, dict]], image_files: list[str],
               sizes: dict[str, tuple[int, int]]):
    """Returns (boxes by image rel path, class names in file order, skipped list)."""
    skipped: list[dict] = []

    # per file: base name -> image id (first entry wins), and id -> file_name
    by_name: list[dict[str, object]] = []
    for _, data in coco_files:
        names: dict[str, object] = {}
        for entry in data["images"]:
            if not isinstance(entry, dict) or not _is_id(entry.get("id")):
                continue
            fn = entry.get("file_name")
            if isinstance(fn, str) and fn:
                names.setdefault(PurePosixPath(fn.replace("\\", "/")).name, entry["id"])
        by_name.append(names)

    # each image file -> (file index, image id) of the deepest COCO file listing its base name
    order = sorted(range(len(coco_files)),
                   key=lambda i: (-len(PurePosixPath(coco_files[i][0]).parts), coco_files[i][0]))
    owner: dict[tuple[int, object], str] = {}
    for rel in image_files:
        base = PurePosixPath(rel).name
        for i in order:
            if _in_scope(rel, _folder(coco_files[i][0])) and base in by_name[i]:
                owner.setdefault((i, by_name[i][base]), rel)
                break

    for i, (json_rel, _) in enumerate(coco_files):
        for base, image_id in by_name[i].items():
            if (i, image_id) not in owner:
                skipped.append(skip(json_rel, f"no image file for '{base}'"))

    boxes: dict[str, list[dict]] = {}
    used: set[str] = set()
    cat_order: list[str] = []
    for i, (json_rel, data) in enumerate(coco_files):
        cats: dict[object, str] = {}
        for c in data["categories"]:
            if isinstance(c, dict) and _is_id(c.get("id")) and isinstance(c.get("name"), str):
                cats.setdefault(c["id"], c["name"])
                cat_order.append(c["name"])
        listed = set(by_name[i].values())
        box_only = 0
        for n, ann in enumerate(data["annotations"]):
            if not isinstance(ann, dict):
                skipped.append(skip(json_rel, f"annotation #{n}: not an object"))
                continue
            label = f"annotation {ann.get('id', '#' + str(n))}"
            image_id = ann.get("image_id")
            if not _is_id(image_id):
                skipped.append(skip(json_rel, f"{label}: bad image_id"))
                continue
            rel = owner.get((i, image_id))
            if rel is None:
                if image_id not in listed:
                    skipped.append(skip(json_rel, f"{label}: unknown image_id {image_id}"))
                continue  # image missing: already reported once for the image
            if rel not in sizes:
                continue  # image would not open: already skipped
            cid = ann.get("category_id")
            if not _is_id(cid) or cid not in cats:
                skipped.append(skip(json_rel, f"{label}: unknown category_id {cid}"))
                continue
            seg = ann.get("segmentation", [])
            polygon = _polygon(seg, *sizes[rel])
            if polygon is not None:
                clipped = polygon_bbox(polygon, *sizes[rel])
            else:
                bbox = _bbox(ann.get("bbox"))
                if bbox is None:
                    skipped.append(skip(json_rel, f"{label}: bad bbox"))
                    continue
                clipped = clip_box(*bbox, *sizes[rel])
                if clipped is None:
                    continue
                if seg != []:
                    box_only += 1
            boxes.setdefault(rel, []).append(
                {"class_name": cats[cid], "bbox": clipped, "polygon": polygon})
            used.add(cats[cid])
        if box_only:
            skipped.append(skip(json_rel, BOX_ONLY_NOTE.format(box_only)))

    class_names = list(dict.fromkeys(n for n in cat_order if n in used))
    return boxes, class_names, skipped
