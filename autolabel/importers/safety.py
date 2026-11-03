"""Path and size checks for uploads, and safe zip extraction (CONTRACT #22)."""

from __future__ import annotations

import re
import zipfile
import zlib
from pathlib import Path

from autolabel.importers.base import ImportRejected, is_image_name, skip

_DRIVE = re.compile(r"^[A-Za-z]:")
_CHUNK = 1 << 20


def normalize_name(name: str) -> str:
    """Use '/' separators (Windows zips and browsers may send '\\')."""
    return name.replace("\\", "/")


def check_names(names: list[str]) -> None:
    """Raise ImportRejected if a name is empty, absolute, has '..' as a path part, or repeats."""
    seen: set[str] = set()
    for raw in names:
        name = normalize_name(raw)
        if not name.strip("/"):
            raise ImportRejected("empty file name in upload")
        if name.startswith("/") or _DRIVE.match(name):
            raise ImportRejected(f"absolute path not allowed: {raw}")
        if ".." in name.split("/"):
            raise ImportRejected(f"'..' not allowed in path: {raw}")
        if name in seen:
            raise ImportRejected(f"path appears twice: {raw}")
        seen.add(name)


def _open_zip(zip_path: Path) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError, ValueError):
        raise ImportRejected("not a valid zip file") from None


def check_zip(zip_path: Path, max_bytes: int) -> int:
    """Check a zip before the job exists. Returns the number of image entries (the job's total)."""
    with _open_zip(zip_path) as zf:
        infos = zf.infolist()
    check_names([i.filename for i in infos])
    total = sum(i.file_size for i in infos)
    if total > max_bytes:
        raise ImportRejected(f"upload too large: {total} bytes unpacked, limit {max_bytes}")
    return sum(1 for i in infos if not i.is_dir() and is_image_name(i.filename))


def extract_zip(zip_path: Path, dest: Path, max_bytes: int) -> list[dict]:
    """Unpack under dest only. A file that grows past its declared size or past the cap
    is stopped, removed and skipped ("too large"). Returns the skipped list."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    skipped: list[dict] = []
    written = 0
    with _open_zip(zip_path) as zf:
        infos = zf.infolist()
        check_names([i.filename for i in infos])
        for info in infos:
            name = normalize_name(info.filename)
            if info.is_dir() or name.endswith("/"):
                continue
            target = (root / name).resolve()
            if root != target and root not in target.parents:  # belt and braces
                raise ImportRejected(f"unsafe path: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            too_large = False
            try:
                with zf.open(info) as src, open(target, "wb") as out:
                    while chunk := src.read(_CHUNK):
                        size += len(chunk)
                        if size > info.file_size or written + size > max_bytes:
                            too_large = True
                            break
                        out.write(chunk)
            except (zipfile.BadZipFile, zlib.error, RuntimeError, NotImplementedError, EOFError):
                # corrupt, encrypted or unsupported entry: skip it, keep the rest
                target.unlink(missing_ok=True)
                skipped.append(skip(name, "cannot unpack"))
                continue
            if too_large:
                target.unlink()
                skipped.append(skip(name, "too large"))
                continue
            written += size
    return skipped
