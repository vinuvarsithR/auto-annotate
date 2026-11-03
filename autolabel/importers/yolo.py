"""YOLO import (CONTRACT #22).

Detected when a `labels/` folder with .txt files sits next to an `images/` folder, and a
`data.yaml` (`names`: list or {index: name}) or `classes.txt` exists; the top-most one is used.
`images/<sub>/<stem>.<ext>` pairs with `labels/<sub>/<stem>.txt` (same parent folder).
Each line is `class cx cy w h`, normalized 0..1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml

from autolabel.importers.base import clip_box, skip

CONFIG_NAMES = ("data.yaml", "classes.txt")


@dataclass
class YoloConfig:
    config: str                  # rel path of the data.yaml / classes.txt used
    names: dict[int, str]        # class index -> name
    label_files: set[str]        # .txt files inside a labels/ folder that has an images/ sibling
    bad_configs: list[str]       # higher config files that could not be read


def _dirs(files: list[str]) -> set[str]:
    out: set[str] = set()
    for rel in files:
        parts = rel.split("/")[:-1]
        for i in range(1, len(parts) + 1):
            out.add("/".join(parts[:i]))
    return out


def _is_label_file(rel: str, dirs: set[str]) -> bool:
    parts = rel.split("/")
    if not rel.lower().endswith(".txt"):
        return False
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "labels":
            return "/".join(parts[:i] + ["images"]) in dirs
    return False


def label_path(image_rel: str) -> str | None:
    """images/<sub>/<stem>.<ext> -> labels/<sub>/<stem>.txt (last `images` folder), else None."""
    parts = image_rel.split("/")
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            stem = PurePosixPath(parts[-1]).stem
            return "/".join(parts[:i] + ["labels"] + parts[i + 1:-1] + [stem + ".txt"])
    return None


def _names_from_yaml(path: Path) -> dict[int, str] | None:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    names = data.get("names") if isinstance(data, dict) else None
    if isinstance(names, list):
        out = {i: str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        out = {int(k): str(v) for k, v in names.items()}
    else:
        return None
    return {i: n for i, n in out.items() if n.strip()} or None


def _names_from_txt(path: Path) -> dict[int, str] | None:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8-sig").splitlines()]
    return {i: n for i, n in enumerate(lines) if n} or None


def find_yolo(root: Path, files: list[str]) -> YoloConfig | None:
    dirs = _dirs(files)
    label_files = {f for f in files if _is_label_file(f, dirs)}
    if not label_files:
        return None
    configs = [f for f in files if PurePosixPath(f).name.lower() in CONFIG_NAMES]
    configs.sort(key=lambda f: (f.count("/"), CONFIG_NAMES.index(PurePosixPath(f).name.lower()), f))
    bad: list[str] = []
    for rel in configs:
        reader = _names_from_yaml if rel.lower().endswith(".yaml") else _names_from_txt
        try:
            names = reader(root / rel)
        except (OSError, ValueError, yaml.YAMLError, TypeError):
            names = None
        if names:
            return YoloConfig(rel, names, label_files, bad)
        bad.append(rel)
    return None


def _parse_label(text: str, names: dict[int, str]):
    """Returns ([(class_name, cx, cy, w, h)], [error reasons])."""
    rows, errors = [], []
    for n, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) != 5:
            errors.append(f"line {n}: expected 5 values, got {len(tokens)}")
            continue
        try:
            nums = [float(t) for t in tokens]
        except ValueError:
            errors.append(f"line {n}: not a number")
            continue
        if not all(math.isfinite(v) for v in nums) or not nums[0].is_integer():
            errors.append(f"line {n}: not a number")
            continue
        cls = int(nums[0])
        if cls not in names:
            errors.append(f"line {n}: unknown class index {cls}")
            continue
        rows.append((names[cls], *nums[1:]))
    return rows, errors


def yolo_boxes(root: Path, cfg: YoloConfig, image_files: list[str],
               sizes: dict[str, tuple[int, int]]):
    """Returns (boxes by image rel path, class names in index order, skipped, used label files)."""
    skipped: list[dict] = [skip(rel, "could not read class names") for rel in cfg.bad_configs]
    parsed: dict[str, list] = {}
    matched: set[str] = set()
    boxes: dict[str, list[dict]] = {}
    used: set[str] = set()
    for rel in image_files:
        lp = label_path(rel)
        if lp is None or lp not in cfg.label_files:
            continue
        matched.add(lp)
        if lp not in parsed:
            try:
                text = (root / lp).read_text(encoding="utf-8-sig")
            except (OSError, UnicodeDecodeError):
                skipped.append(skip(lp, "cannot read label file"))
                parsed[lp] = []
                continue
            rows, errors = _parse_label(text, cfg.names)
            skipped.extend(skip(lp, e) for e in errors)
            parsed[lp] = rows
        if rel not in sizes:
            continue
        width, height = sizes[rel]
        for name, cx, cy, w, h in parsed[lp]:
            clipped = clip_box((cx - w / 2) * width, (cy - h / 2) * height,
                               w * width, h * height, width, height)
            if clipped is None:
                continue
            boxes.setdefault(rel, []).append({"class_name": name, "bbox": clipped})
            used.add(name)

    for lp in sorted(cfg.label_files - matched):
        skipped.append(skip(lp, "no matching image"))
    class_names = list(dict.fromkeys(cfg.names[i] for i in sorted(cfg.names) if cfg.names[i] in used))
    return boxes, class_names, skipped, cfg.label_files | {cfg.config}
