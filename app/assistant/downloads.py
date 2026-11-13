"""
downloads.py - files a tool made for the human (#35): today only the make_train_package zip.

data/assistant/downloads/<token>.zip  + <token>.json {"file_name", "created_at", "expires_at"}
token = secrets.token_urlsafe(16). A file is served for 24 h; older ones are deleted at startup
and each time a new package is saved.
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path

LIFETIME = timedelta(hours=24)
_TOKEN = re.compile(r"^[A-Za-z0-9_-]{8,64}$")  # token_urlsafe alphabet; also blocks ../ paths
_FMT = "%Y-%m-%dT%H:%M:%SZ"


def folder(data_dir: str | Path) -> Path:
    return Path(data_dir) / "assistant" / "downloads"


def _parse(text: str) -> datetime | None:
    try:
        return datetime.strptime(text, _FMT).replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _expired(meta_path: Path, now: datetime) -> bool:
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        expires = _parse(meta.get("expires_at"))
    except (OSError, ValueError, AttributeError):
        expires = None
    if expires is None:  # unreadable metadata: fall back to the file's age
        try:
            mtime = datetime.fromtimestamp(meta_path.stat().st_mtime, timezone.utc)
        except OSError:
            return True
        expires = mtime + LIFETIME
    return now >= expires


def cleanup(data_dir: str | Path, now: datetime | None = None) -> int:
    """Delete every download older than 24 h (and zips without metadata). Returns how many."""
    d = folder(data_dir)
    if not d.is_dir():
        return 0
    now = now or datetime.now(timezone.utc)
    removed = 0
    for zip_path in d.glob("*.zip"):
        meta = zip_path.with_suffix(".json")
        if not meta.exists() or _expired(meta, now):
            zip_path.unlink(missing_ok=True)
            meta.unlink(missing_ok=True)
            removed += 1
    for meta in d.glob("*.json"):  # metadata whose zip is gone
        if not meta.with_suffix(".zip").exists():
            meta.unlink(missing_ok=True)
    return removed


def save(data_dir: str | Path, blob: bytes, file_name: str) -> tuple[str, str]:
    """Store the zip; returns (token, expires_at)."""
    cleanup(data_dir)
    d = folder(data_dir)
    d.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(16)
    now = datetime.now(timezone.utc)
    expires_at = (now + LIFETIME).strftime(_FMT)
    (d / f"{token}.zip").write_bytes(blob)
    (d / f"{token}.json").write_text(json.dumps({
        "file_name": file_name, "created_at": now.strftime(_FMT), "expires_at": expires_at}),
        encoding="utf-8")
    return token, expires_at


def find(data_dir: str | Path, token: str) -> tuple[Path, str] | None:
    """(zip path, file name) of a live download, or None (unknown, malformed or expired)."""
    if not _TOKEN.match(token or ""):
        return None
    d = folder(data_dir)
    zip_path, meta = d / f"{token}.zip", d / f"{token}.json"
    if not zip_path.is_file() or not meta.is_file():
        return None
    if _expired(meta, datetime.now(timezone.utc)):
        zip_path.unlink(missing_ok=True)
        meta.unlink(missing_ok=True)
        return None
    try:
        file_name = json.loads(meta.read_text(encoding="utf-8"))["file_name"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return zip_path, str(file_name)
