"""
formats.py - COCO and YOLO export (CONTRACT #14) as pure functions on the API JSON
shapes, plus the zip builder.

Zip layouts
-----------
COCO  (<project>_coco.zip)
    annotations.json          one COCO file for all exported images
    images/<file_name>        only with include_images
YOLO  (<project>_yolo.zip)
    data.yaml                 nc + names (class index = position in the class list)
    classes.txt               one class name per line, same order
    labels/<stem>.txt         one per exported image; empty file = no boxes
    images/<file_name>        only with include_images
YOLO-seg  (<project>_yolo-seg.zip)
    same layout as YOLO; each line is `class x1 y1 x2 y2 ...` (a plain box becomes its 4 corners)

Rows with a polygon: COCO `segmentation` holds it in pixels and `area` is the polygon's area;
plain YOLO writes the polygon's bbox.

If two exported images share a stem (a.jpg and a.png), the later ones get the image id
as a prefix (`<id>_a.png`, `labels/<id>_a.txt`) so files never overwrite each other.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import PurePosixPath
from typing import Callable

from autolabel.importers.base import polygon_area
from autolabel.export.select import (
    NothingToExport,
    boxes_for_image,
    exportable_images,
    validate_statuses,
)

FORMATS = ("coco", "yolo", "yolo-seg")


def _class_index(classes: list[dict]) -> dict[int, int]:
    """class_id -> position in the project class list."""
    return {c["id"]: i for i, c in enumerate(classes)}


def export_names(images: list[dict]) -> dict[int, str]:
    """image id -> file name used inside the zip (unique stems, see module doc)."""
    names: dict[int, str] = {}
    seen: set[str] = set()
    for im in images:
        name = im["file_name"]
        if PurePosixPath(name).stem in seen:
            name = f"{im['id']}_{name}"
        seen.add(PurePosixPath(name).stem)
        names[im["id"]] = name
    return names


def coco_polygon(polygon: list[list[float]], width: int, height: int) -> tuple[list, float]:
    """Normalized polygon -> (COCO `segmentation` in pixels, area in pixels)."""
    points = [[x * width, y * height] for x, y in polygon]
    return [[v for p in points for v in p]], polygon_area(points)


def to_coco(
    images: list[dict],
    classes: list[dict],
    annotations: list[dict],
    statuses: list[str] | None = None,
) -> dict:
    """
    Build one COCO dict. Only done/empty images, only boxes in `statuses`
    (default: approved), never `rejected`. Category ids are 1-based, in project
    class order. Image and annotation ids are the app ids. A mask (COCO RLE) goes
    into `segmentation`; boxes without a mask get `[]`. A polygon comes before the mask.
    """
    statuses = validate_statuses(statuses)
    images = exportable_images(images)
    idx = _class_index(classes)
    names = export_names(images)

    out_images, out_anns = [], []
    for im in images:
        out_images.append(
            {"id": im["id"], "file_name": names[im["id"]], "width": im["width"], "height": im["height"]}
        )
        for a in boxes_for_image(im, annotations, statuses):
            if a["class_id"] not in idx:
                raise ValueError(f"annotation {a['id']} has unknown class_id {a['class_id']}")
            x, y, w, h = (float(v) for v in a["bbox"])
            if a.get("polygon"):
                segmentation, area = coco_polygon(a["polygon"], im["width"], im["height"])
            else:
                segmentation, area = a["mask"] if a.get("mask") else [], w * h
            out_anns.append(
                {
                    "id": a["id"],
                    "image_id": im["id"],
                    "category_id": idx[a["class_id"]] + 1,
                    "bbox": [x, y, w, h],
                    "area": area,
                    "iscrowd": 0,
                    "segmentation": segmentation,
                }
            )

    return {
        "info": {"description": "auto-annotate export"},
        "images": out_images,
        "categories": [{"id": i + 1, "name": c["name"]} for i, c in enumerate(classes)],
        "annotations": out_anns,
    }


def _yolo_line(cls: int, bbox: list[float], width: int, height: int) -> str | None:
    """One YOLO line (class cx cy w h, normalised). The box is clipped to the image;
    None if nothing is left."""
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = max(0.0, x), max(0.0, y)
    x1, y1 = min(float(width), x + w), min(float(height), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    cx, cy = (x0 + x1) / 2 / width, (y0 + y1) / 2 / height
    nw, nh = (x1 - x0) / width, (y1 - y0) / height
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _seg_line(cls: int, annotation: dict, width: int, height: int) -> str | None:
    """One YOLO-seg line. A row without a polygon uses the 4 corners of its clipped bbox;
    None if nothing is left."""
    polygon = annotation.get("polygon")
    if not polygon:
        x, y, w, h = (float(v) for v in annotation["bbox"])
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(width), x + w), min(float(height), y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        polygon = [[x0 / width, y0 / height], [x1 / width, y0 / height],
                   [x1 / width, y1 / height], [x0 / width, y1 / height]]
    return f"{cls} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)


def to_yolo(
    images: list[dict],
    classes: list[dict],
    annotations: list[dict],
    statuses: list[str] | None = None,
    seg: bool = False,
) -> dict[str, str]:
    """
    Build the YOLO text files: {path in zip: file text}. Same image/box rules as
    to_coco. Every exported image gets a label file (empty for no boxes). `seg=True`
    writes YOLO-seg lines.
    """
    statuses = validate_statuses(statuses)
    images = exportable_images(images)
    idx = _class_index(classes)
    names = export_names(images)

    class_names = [c["name"] for c in classes]
    files: dict[str, str] = {
        "data.yaml": "\n".join(
            [
                "path: .",
                "train: images",
                "val: images",
                f"nc: {len(class_names)}",
                "names: " + json.dumps(class_names),
                "",
            ]
        ),
        "classes.txt": "".join(n + "\n" for n in class_names),
    }
    for im in images:
        lines = []
        for a in boxes_for_image(im, annotations, statuses):
            if a["class_id"] not in idx:
                raise ValueError(f"annotation {a['id']} has unknown class_id {a['class_id']}")
            if seg:
                line = _seg_line(idx[a["class_id"]], a, im["width"], im["height"])
            else:
                line = _yolo_line(idx[a["class_id"]], a["bbox"], im["width"], im["height"])
            if line:
                lines.append(line)
        stem = PurePosixPath(names[im["id"]]).stem
        files[f"labels/{stem}.txt"] = "".join(l + "\n" for l in lines)
    return files


def to_yolo_seg(
    images: list[dict],
    classes: list[dict],
    annotations: list[dict],
    statuses: list[str] | None = None,
) -> dict[str, str]:
    """to_yolo with YOLO-seg lines (CONTRACT #14)."""
    return to_yolo(images, classes, annotations, statuses, seg=True)


def build_zip(
    fmt: str,
    images: list[dict],
    classes: list[dict],
    annotations: list[dict],
    statuses: list[str] | None = None,
    image_bytes: Callable[[dict], bytes] | None = None,
) -> bytes:
    """
    Return the zip as bytes. `image_bytes(image) -> bytes` is given only when the
    image files should go in the zip (include_images). Raises ValueError for a bad
    format/statuses and NothingToExport when no image is done/empty.
    """
    if fmt not in FORMATS:
        raise ValueError(f"format must be one of {list(FORMATS)}")
    statuses = validate_statuses(statuses)
    chosen = exportable_images(images)
    if not chosen:
        raise NothingToExport("no image has review_status done or empty")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if fmt == "coco":
            coco = to_coco(chosen, classes, annotations, statuses)
            zf.writestr("annotations.json", json.dumps(coco, indent=1))
        else:
            make = to_yolo_seg if fmt == "yolo-seg" else to_yolo
            for path, text in make(chosen, classes, annotations, statuses).items():
                zf.writestr(path, text)
        if image_bytes is not None:
            names = export_names(chosen)
            for im in chosen:
                zf.writestr(f"images/{names[im['id']]}", image_bytes(im))
    return buf.getvalue()
