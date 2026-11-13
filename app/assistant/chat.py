"""
chat.py - the sidebar chat's Claude tool-use loop (docs/CONTRACT.md "Sidebar chat", #32 - #34).

A manual loop over messages.create (not the beta tool runner): exact tool-call cap,
server-held confirmations, and a fake client in tests.

| Step       | Rule |
| Request    | client.beta.messages.create(model, max_tokens, system=SYSTEM, tools=CHAT_TOOLS, messages,
|            | output_config={"effort"}, cache_control={"type": "ephemeral"}, betas + fallbacks="default")
|            | for the models in FALLBACK_MODELS; other models: client.messages.create without the two.
| Prefix     | SYSTEM and CHAT_TOOLS are constants (same on every request of every session: cache).
| History    | append-only: each response's full content, then one user message with all tool_results.
| Tools      | in order; a `confirm` Claude sends is dropped (only the Confirm button confirms);
|            | needs_confirmation -> a server-held pending item.
| Cap        | after max_tool_calls calls, further tool_use blocks get an error tool_result
|            | (outcome not_run) and Claude is not called again: stopped tool_limit.
| Memory     | in this process only: lost on restart. 200 sessions, 40 user messages each.

The API key is never logged or returned; error texts are scrubbed.
"""

from __future__ import annotations

import json
import logging
import secrets as _secrets
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from app.api.errors import ApiError
from app.assistant.settings import AssistantSettings, scrub
from app.assistant.tools import TOOLS, ToolContext, call_tool, outcome_of

log = logging.getLogger("app.assistant")

MAX_SESSIONS = 200
MAX_USER_MESSAGES = 40
FALLBACK_MODELS = ("claude-sonnet-5-5", "claude-opus-5-5", "claude-opus-5",
                   "claude-fable-5-1", "claude-fable-5-2")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
LIMIT_MESSAGE = "tool call limit reached for this message"
MAX_TOKENS_MESSAGE = "not run: the reply hit max_tokens before this tool could run"
REFUSAL_REPLY = "Claude declined this request."

SYSTEM = (
    "You work on an auto-annotation app for object detection through the tools you are given. "
    "Projects have images, classes and boxes; a model proposes boxes and humans review them. "
    "Boxes are [x, y, w, h] in image pixels; polygons are normalized 0..1. "
    "Use the read tools to look before you change anything, and keep answers short. "
    "In this chat a risky action is confirmed only by the user's Confirm button, never by you: "
    "when a tool result says needs_confirmation, tell the user in one or two sentences what will "
    "happen and ask them to press Confirm. Never say such an action is done until a later note "
    "says the user pressed Confirm and it returned ok. The first line of each user message says "
    "which project the user has open and who they are.")


def _chat_tool(t: dict) -> dict:
    schema = json.loads(json.dumps(t["input_schema"]))
    schema.get("properties", {}).pop("confirm", None)
    if "required" in schema:
        schema["required"] = [r for r in schema["required"] if r != "confirm"]
    return {"name": t["name"], "description": t["description"], "input_schema": schema}


CHAT_TOOLS: list[dict] = [_chat_tool(t) for t in TOOLS]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a field from an SDK object or a plain dict (the fake client returns dicts)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


@dataclass
class Pending:
    confirm_id: str
    tool: str
    effect: str
    arguments: dict

    def public(self) -> dict:
        return {"confirm_id": self.confirm_id, "tool": self.tool, "effect": self.effect,
                "arguments": self.arguments}


@dataclass
class Session:
    id: str
    history: list = field(default_factory=list)
    pending: dict = field(default_factory=dict)  # confirm_id -> Pending
    user_messages: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class ChatError(Exception):
    """The Claude API failed: becomes 502 assistant_error with an already scrubbed message."""


def _append_user(history: list, blocks: list) -> None:
    """Add a user turn; merged into the last message when that is already a user turn."""
    if history and history[-1]["role"] == "user":
        history[-1]["content"].extend(blocks)
    else:
        history.append({"role": "user", "content": list(blocks)})


class ChatManager:
    def __init__(self, state: Any, settings: AssistantSettings, client_factory):
        self.state = state
        self.settings = settings
        self._client_factory = client_factory  # () -> anthropic client (built lazily)
        self._client = None
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ sessions

    @property
    def client(self):
        if self._client is None:
            self._client = self._client_factory()
        return self._client

    def get(self, session_id: str) -> Session:
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise ApiError(404, "not_found", f"chat session {session_id} not found: chats are "
                           "kept in the server's memory and are lost when it restarts")
        return s

    def new(self) -> Session:
        s = Session(id=str(uuid.uuid4()))
        with self._lock:
            self._sessions[s.id] = s
            while len(self._sessions) > MAX_SESSIONS:  # oldest dropped
                self._sessions.popitem(last=False)
        return s

    def forget(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                raise ApiError(404, "not_found", f"chat session {session_id} not found: chats "
                               "are kept in the server's memory and are lost when it restarts")

    @staticmethod
    def _acquire(s: Session) -> None:
        if not s.lock.acquire(blocking=False):
            raise ApiError(409, "conflict", "this chat is still answering the previous message")

    # ------------------------------------------------------------ #32 one message

    def send(self, session_id: str | None, project: dict, user: str, message: str) -> dict:
        s = self.get(session_id) if session_id else None
        if s is not None:
            self._acquire(s)
        else:
            s = self.new()
            s.lock.acquire()
        try:
            if s.user_messages >= MAX_USER_MESSAGES:
                raise ApiError(409, "conflict", f"this chat has {MAX_USER_MESSAGES} messages; "
                               "start a new chat")
            s.pending.clear()  # pending items live until the next message
            return self._run(s, project, user, message)
        finally:
            s.lock.release()

    def _request(self, history: list) -> Any:
        st = self.settings
        kwargs = dict(model=st.model, max_tokens=st.max_tokens, system=SYSTEM, tools=CHAT_TOOLS,
                      messages=list(history), output_config={"effort": st.effort},
                      cache_control={"type": "ephemeral"})
        if st.model in FALLBACK_MODELS:
            return self.client.beta.messages.create(**kwargs, betas=[FALLBACK_BETA],
                                                    fallbacks="default")
        return self.client.messages.create(**kwargs)

    def _api_error(self, exc: Exception) -> ChatError:
        secrets = self.settings.secrets()
        status = getattr(exc, "status_code", None)
        if status is None and type(exc).__name__ in ("APIConnectionError", "APITimeoutError"):
            text = "Claude API unreachable"
        elif status is None:
            text = "Claude API error: " + scrub(f"{type(exc).__name__}: {exc}", secrets)
        else:
            detail = getattr(exc, "message", None) or str(exc)
            text = f"Claude API error {status}: {scrub(str(detail), secrets)}"
        text = scrub(text.splitlines()[0][:500], secrets)
        log.warning("assistant chat: %s", text)
        return ChatError(text)

    def _run(self, s: Session, project: dict, user: str, message: str) -> dict:
        history = s.history
        # undo point: everything this message adds is removed on refusal or API error
        mark = len(history)
        merged_from = len(history[-1]["content"]) if history and history[-1]["role"] == "user" \
            else None

        def rollback():
            if merged_from is not None:
                del history[mark:]
                del history[-1]["content"][merged_from:]
            else:
                del history[mark:]

        _append_user(history, [{"type": "text", "text":
                                f"[project {project['id']} \"{project['name']}\"; user {user}]\n"
                                f"{message}"}])
        s.user_messages += 1
        ctx = ToolContext(self.state, "chat", user)
        usage = {"input_tokens": 0, "output_tokens": 0}
        tool_calls: list[dict] = []
        pending: list[Pending] = []
        calls_made = 0
        stopped = "end_turn"
        last_content: list = []

        while True:
            try:
                response = self._request(history)
            except Exception as exc:
                rollback()
                s.user_messages -= 1
                raise self._api_error(exc) from None
            u = _get(response, "usage")
            for k in usage:
                usage[k] += int(_get(u, k, 0) or 0)
            content = list(_get(response, "content", []) or [])
            last_content = content
            stop = _get(response, "stop_reason")

            if stop == "refusal":
                rollback()  # the user turn (and anything after it) leaves the history
                s.user_messages -= 1
                s.pending.clear()  # their tool_use turns are gone from the history too
                return self._response(s, REFUSAL_REPLY, tool_calls, [], "refusal", usage)

            history.append({"role": "assistant", "content": content})
            if stop == "pause_turn":
                continue
            uses = [b for b in content if _get(b, "type") == "tool_use"]

            if stop == "tool_use" and uses:
                results, limit_hit = [], False
                for b in uses:
                    args = dict(_get(b, "input", {}) or {})
                    args.pop("confirm", None)  # only the Confirm button confirms
                    if calls_made >= self.settings.max_tool_calls:
                        limit_hit = True
                        tool_calls.append({"tool": _get(b, "name"), "arguments": args,
                                           "outcome": "not_run"})
                        results.append({"type": "tool_result", "tool_use_id": _get(b, "id"),
                                        "content": LIMIT_MESSAGE, "is_error": True})
                        continue
                    calls_made += 1
                    envelope = call_tool(ctx, _get(b, "name"), args)
                    outcome = outcome_of(envelope)
                    tool_calls.append({"tool": _get(b, "name"), "arguments": args,
                                       "outcome": outcome})
                    if outcome == "needs_confirmation":
                        nc = envelope["needs_confirmation"]
                        item = Pending(_secrets.token_urlsafe(8), nc["tool"], nc["effect"],
                                       nc["arguments"])
                        s.pending[item.confirm_id] = item
                        pending.append(item)
                    results.append({"type": "tool_result", "tool_use_id": _get(b, "id"),
                                    "content": json.dumps(envelope),
                                    "is_error": outcome == "error"})
                _append_user(history, results)
                if limit_hit:
                    stopped = "tool_limit"
                    break
                continue

            if stop == "max_tokens":
                stopped = "max_tokens"
                if uses:  # keep the stored history valid: every tool_use gets a result
                    _append_user(history, [{"type": "tool_result", "tool_use_id": _get(b, "id"),
                                            "content": MAX_TOKENS_MESSAGE, "is_error": True}
                                           for b in uses])
                break
            stopped = "end_turn"
            break

        reply = "\n\n".join(_get(b, "text", "") for b in last_content
                            if _get(b, "type") == "text")
        return self._response(s, reply, tool_calls, pending, stopped, usage)

    @staticmethod
    def _response(s: Session, reply: str, tool_calls: list, pending: list, stopped: str,
                  usage: dict) -> dict:
        return {"session_id": s.id, "reply": reply, "tool_calls": tool_calls,
                "pending": [p.public() for p in pending], "stopped": stopped, "usage": usage}

    # ------------------------------------------------------------ #33 confirm / cancel

    def confirm(self, session_id: str, confirm_id: str, decision: str, user: str) -> dict:
        s = self.get(session_id)
        self._acquire(s)
        try:
            item = s.pending.pop(confirm_id, None)
            if item is None:
                raise ApiError(409, "conflict", f"confirm_id {confirm_id} is unknown, already "
                               "used or expired (a newer message was sent)")
            args_text = json.dumps(item.arguments)
            if decision == "confirm":
                ctx = ToolContext(self.state, "chat", user, button=True)
                result = call_tool(ctx, item.tool, {**item.arguments, "confirm": True})
                note = (f"[{user} pressed Confirm: {item.tool} {args_text} -> "
                        f"{outcome_of(result)}]")
            else:
                result = None
                note = f"[{user} pressed Cancel: {item.tool} {args_text} -> not run]"
            _append_user(s.history, [{"type": "text", "text": note}])
            return {"confirm_id": confirm_id, "tool": item.tool, "decision": decision,
                    "result": result}
        finally:
            s.lock.release()
