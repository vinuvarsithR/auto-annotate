"""
test_api_import_stub.py - (helper, no tests) a small stand-in for autolabel.importers (export's code, CONTRACT #22)
so the backend tests run before export merges, and stay independent of it afterwards.

Same names as the contract interface. The stub's only label format: a `labels.json`
file at the root, {"format": "coco", "boxes": {"rel/path.jpg": [{"class_name", "bbox"}]}}.
No labels.json -> format "images". Every other non-image file is skipped "not an image".
"""

from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


class ImportRejected(ValueError):
    pass


def check_names(names: list[str]) -> None:
    seen = set()
    for n in names:
        parts = n.replace("\\", "/").split("/")
        if ".." in parts or n.startswith(("/", "\\")) or PureWindowsPath(n).drive:
            raise ImportRejected(f"unsafe path in upload: {n}")
        if n in seen:
            raise ImportRejected(f"repeated path in upload: {n}")
        seen.add(n)


def check_zip(zip_path: Path, max_bytes: int) -> int:
    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError):
        raise ImportRejected("not a zip file")
    with zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        try:
            check_names([i.filename for i in infos])
        except ImportRejected as exc:
            raise ImportRejected(str(exc).replace("upload", "zip"))
        if sum(i.file_size for i in infos) > max_bytes:
            raise ImportRejected("zip is too large")
        return sum(1 for i in infos if PurePosixPath(i.filename).suffix.lower() in IMAGE_EXTS)


def extract_zip(zip_path: Path, dest: Path, max_bytes: int) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for i in zf.infolist():
            if not i.is_dir():
                zf.extract(i, dest)
    return []


@dataclass(frozen=True)
class ImportItem:
    rel_path: str
    boxes: list[dict] = field(default_factory=list)


@dataclass
class ImportPlan:
    format: str
    items: list[ImportItem]
    class_names: list[str]
    skipped: list[dict]


def plan_import(root: Path) -> ImportPlan:
    import json
    labels_file = root / "labels.json"
    labels = json.loads(labels_file.read_text()) if labels_file.is_file() else None
    boxes = (labels or {}).get("boxes", {})
    items, skipped, names = [], [], []
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix()
        if rel == "labels.json":
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            skipped.append({"file_name": rel, "reason": "not an image"})
            continue
        b = boxes.get(rel, [])
        names += [x["class_name"] for x in b]
        items.append(ImportItem(rel, b))
    return ImportPlan(labels["format"] if labels else "images", items,
                      list(dict.fromkeys(names)), skipped)


def module() -> SimpleNamespace:
    """What load_importers() returns: an object with the contract's names."""
    return SimpleNamespace(IMAGE_EXTS=IMAGE_EXTS, ImportRejected=ImportRejected,
                           check_names=check_names, check_zip=check_zip,
                           extract_zip=extract_zip, plan_import=plan_import)
