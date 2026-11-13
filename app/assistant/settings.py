"""
settings.py - Claude connector settings from .env (docs/CONTRACT.md "Sidebar chat").

Read once at app startup, after load_dotenv(REPO_ROOT / ".env", override=False), so a variable
already set in the environment wins over .env.

ANTHROPIC_API_KEY and AUTOLABEL_MCP_TOKEN are secrets: they are only ever compared or passed to
the SDK. Nothing here logs, prints or returns them; repr() hides them too.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("app.assistant")

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"  # tests point this at a temp path

DEFAULT_MODEL = "claude-sonnet-5-5"
DEFAULT_EFFORT = "low"
DEFAULT_MAX_TOOL_CALLS = 10
DEFAULT_MAX_TOKENS = 16000


@dataclass(frozen=True)
class AssistantSettings:
    api_key: str = field(default="", repr=False)      # secret
    mcp_token: str = field(default="", repr=False)    # secret
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS
    max_tokens: int = DEFAULT_MAX_TOKENS

    @property
    def chat_available(self) -> bool:
        return bool(self.api_key)

    @property
    def token_required(self) -> bool:
        return bool(self.mcp_token)

    def secrets(self) -> list[str]:
        """Texts that must never leave the server (for scrubbing)."""
        return [s for s in (self.api_key, self.mcp_token) if s]


def _positive_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        log.warning("%s must be a whole number >= 1; using %s", name, default)
        return default
    return value


def load_settings(env_file: Path | None = None) -> AssistantSettings:
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file or ENV_FILE, override=False)
    except ImportError:  # python-dotenv is in requirements; plain env vars still work
        pass
    return AssistantSettings(
        api_key=(os.environ.get("ANTHROPIC_API_KEY") or "").strip(),
        mcp_token=(os.environ.get("AUTOLABEL_MCP_TOKEN") or "").strip(),
        model=(os.environ.get("AUTOLABEL_CLAUDE_MODEL") or "").strip() or DEFAULT_MODEL,
        effort=(os.environ.get("AUTOLABEL_CLAUDE_EFFORT") or "").strip() or DEFAULT_EFFORT,
        max_tool_calls=_positive_int("AUTOLABEL_CLAUDE_MAX_TOOL_CALLS", DEFAULT_MAX_TOOL_CALLS),
        max_tokens=_positive_int("AUTOLABEL_CLAUDE_MAX_TOKENS", DEFAULT_MAX_TOKENS),
    )


def scrub(text: str, secrets) -> str:
    """Replace every secret's text by [redacted]."""
    for s in secrets:
        if s:
            text = text.replace(s, "[redacted]")
    return text
