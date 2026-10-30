"""
gold.py - load a gold set folder:

    gold_sets/<name>/
        classes.yaml       class names, descriptions, optional aliases
        annotations.json   hand-checked boxes in COCO format
        images/            the image files

Category names in annotations.json are mapped to canonical names using the
'aliases' in classes.yaml (e.g. 'Person' and 'a_person' -> 'person').
Categories that match nothing in classes.yaml (e.g. Roboflow's project-name
category, or junk like '2') are skipped and reported.

Images are identified by file_name everywhere, so predictions and gold
line up even if two exports number their images differently.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from autolabel.evaluate import Box

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Example:
    image: Path  # absolute path
    bbox: tuple[float, float, float, float]  # x, y, w, h in pixels


@dataclass
class ClassSpec:
    name: str
    description: str
    aliases: list[str] = field(default_factory=list)
    examples: list[Example] = field(default_factory=list)  # set -> image-guided query instead of text


@dataclass
class GoldSet:
    classes: list[ClassSpec]
    boxes: list[Box]
    image_files: list[str]  # file names that have an entry in annotations.json
    skipped_categories: Counter  # category name -> number of boxes skipped


def load_classes(path: str | Path) -> list[ClassSpec]:
    """
    Read classes.yaml. Every class needs a name and a description.
    Optional 'examples': [{image: path relative to the yaml, bbox: [x, y, w, h]}].
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    specs = []
    for i, c in enumerate(data.get("classes", [])):
        name = str(c.get("name", "")).strip()
        desc = str(c.get("description", "")).strip()
        if not name:
            raise ValueError(f"{path}: class #{i + 1} has no name")
        if not desc or desc.upper().startswith("TODO"):
            raise ValueError(f"{path}: class '{name}' needs a description (it becomes the OWLv2 prompt)")
        aliases = [str(a) for a in c.get("aliases", []) or []]
        examples = []
        for j, e in enumerate(c.get("examples", []) or []):
            bbox = e.get("bbox") if isinstance(e, dict) else None
            if not isinstance(e, dict) or "image" not in e or not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"{path}: class '{name}' example #{j + 1} needs 'image' and 'bbox: [x, y, w, h]'")
            x, y, w, h = (float(v) for v in bbox)
            if w <= 0 or h <= 0:
                raise ValueError(f"{path}: class '{name}' example #{j + 1} has an empty bbox")
            examples.append(Example((Path(path).parent / e["image"]).resolve(), (x, y, w, h)))
        specs.append(ClassSpec(name, desc, aliases, examples))
    if not specs:
        raise ValueError(f"{path}: no classes found")
    names = [s.name for s in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"{path}: duplicate class names")
    return specs


def alias_map(classes: list[ClassSpec]) -> dict[str, str]:
    """Map every accepted spelling (case-insensitive) to its canonical name."""
    m: dict[str, str] = {}
    for c in classes:
        for spelling in [c.name, *c.aliases]:
            key = spelling.strip().lower()
            if key in m and m[key] != c.name:
                raise ValueError(f"'{spelling}' is listed under both '{m[key]}' and '{c.name}'")
            m[key] = c.name
    return m


def load_gold_set(folder: str | Path, classes_path: str | Path | None = None) -> GoldSet:
    """classes_path: use this classes.yaml instead of the gold set's own (e.g. to try other prompts)."""
    folder = Path(folder)
    classes = load_classes(classes_path or folder / "classes.yaml")
    names = alias_map(classes)

    coco = json.loads((folder / "annotations.json").read_text(encoding="utf-8"))
    cat_name = {c["id"]: c["name"] for c in coco.get("categories", [])}
    img_file = {im["id"]: im["file_name"] for im in coco.get("images", [])}

    boxes: list[Box] = []
    skipped: Counter = Counter()
    for a in coco.get("annotations", []):
        raw = cat_name.get(a["category_id"], f"<unknown id {a['category_id']}>")
        canonical = names.get(raw.strip().lower())
        if canonical is None:
            skipped[raw] += 1
            continue
        x, y, w, h = (float(v) for v in a["bbox"])
        boxes.append(Box(img_file[a["image_id"]], canonical, (x, y, w, h), 1.0))

    return GoldSet(classes, boxes, sorted(img_file.values()), skipped)


def list_images(folder: str | Path) -> list[Path]:
    """All image files directly inside a folder, sorted by name."""
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
