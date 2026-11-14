"""
users.py - per-teammate MCP tokens (docs/CONTRACT.md "#36 change: per-teammate tokens").

The users file (AUTOLABEL_MCP_USERS_FILE, default <data dir>/mcp_users.json) stores HASHES only:
    {"users": [{"email": "...", "token_sha256": "64 hex", "added_at": "...Z"}]}
A token is "alu_" + secrets.token_urlsafe(32); only scripts/add_teammate.py ever sees it, once.

* UsersFile: what the MCP gate reads. Re-read when the file's mtime or size changes, so adding or
  removing a teammate needs no restart. Missing file = no users. Unreadable / bad JSON -> logged
  once (the path only, never the content), treated as no users.
* add_user / remove_user / list_users: what add_teammate.py calls; writes are atomic (temp file
  in the same folder + os.replace).
Nothing here logs, prints or returns a token.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("app.assistant")

TOKEN_PREFIX = "alu_"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_token() -> str:
    return TOKEN_PREFIX + secrets.token_urlsafe(32)


def clean_email(raw: str) -> str | None:
    """The email rule: one '@', a dot after it, <= 254 chars; lower-cased. None if it fails."""
    if not isinstance(raw, str):
        return None
    email = raw.strip().lower()
    if not email or len(email) > 254 or email.count("@") != 1 or any(c.isspace() for c in email):
        return None
    local, domain = email.split("@")
    if not local or "." not in domain or domain.startswith(".") or domain.endswith("."):
        return None
    return email


def _parse(text: str) -> list[dict]:
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("users"), list):
        raise ValueError("expected {\"users\": [...]}")
    users = []
    for u in data["users"]:
        if (not isinstance(u, dict) or not isinstance(u.get("email"), str)
                or not isinstance(u.get("token_sha256"), str)
                or not _HEX64.match(u["token_sha256"])):
            raise ValueError("bad user entry")
        users.append({"email": u["email"].lower(), "token_sha256": u["token_sha256"],
                      "added_at": u.get("added_at")})
    return users


class UsersFile:
    """The gate's view of the users file, re-read when it changes."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._key = None          # (mtime_ns, size) of the version in _users; None = missing
        self._users: list[dict] = []
        self._warned_key = None
        self._lock = threading.Lock()

    def users(self) -> list[dict]:
        with self._lock:
            try:
                st = self.path.stat()
            except OSError:
                self._key, self._users = None, []
                return []
            key = (st.st_mtime_ns, st.st_size)
            if key != self._key:
                try:
                    self._users = _parse(self.path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    self._users = []
                    if self._warned_key != key:  # once per broken version; path only
                        log.warning("MCP users file %s is unreadable or not valid; "
                                    "treating it as no users", self.path)
                        self._warned_key = key
                self._key = key
            return list(self._users)

    def count(self) -> int:
        return len(self.users())

    def match(self, token: str) -> str | None:
        """The email whose token hash matches `token`, or None."""
        if not token:
            return None
        given = token_hash(token)
        found = None
        for u in self.users():  # no early exit: same work whichever entry matches
            if hmac.compare_digest(given, u["token_sha256"]):
                found = u["email"]
        return found


# ------------------------------------------------------------------ writing (add_teammate.py)

def _read_for_write(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return _parse(path.read_text(encoding="utf-8"))


def _write(path: Path, users: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".mcp_users_", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"users": users}, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def add_user(path: str | Path, email: str) -> str:
    """Store a new token's hash for `email` (already cleaned); an existing entry gets the new
    token (the old one stops working). Returns the token: the only place it ever exists."""
    path = Path(path)
    users = [u for u in _read_for_write(path) if u["email"] != email]
    token = new_token()
    users.append({"email": email, "token_sha256": token_hash(token),
                  "added_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    _write(path, users)
    return token


def remove_user(path: str | Path, email: str) -> bool:
    path = Path(path)
    users = _read_for_write(path)
    keep = [u for u in users if u["email"] != email]
    if len(keep) == len(users):
        return False
    _write(path, keep)
    return True


def list_users(path: str | Path) -> list[dict]:
    """email + added_at only."""
    return [{"email": u["email"], "added_at": u["added_at"]} for u in _read_for_write(Path(path))]
