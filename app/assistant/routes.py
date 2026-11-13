"""
routes.py - endpoints 31-35 under /api/assistant (docs/CONTRACT.md "Sidebar chat").

| #  | Route                                  | Notes |
| 31 | GET /assistant/status                  | never contains a key or token |
| 32 | POST /assistant/chat                   | errors: 503 no key, 400 body, 404 project, 404 session, 409, 502 |
| 33 | POST /assistant/chat/confirm           | runs the stored pending tool (button = the only confirm) |
| 34 | DELETE /assistant/chat/{session_id}    | 204 |
| 35 | GET /assistant/downloads/{token}       | the make_train_package zip, 24 h |

Bodies are read as plain JSON and checked here, so "no key" (503) comes before a bad body (400).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from fastapi import APIRouter, Body, Depends, Request, Response
from fastapi.responses import FileResponse

from app.api.deps import get_conn, require_project
from app.api.errors import ApiError
from app.assistant import downloads
from app.assistant.chat import ChatError
from app.assistant.tools import clean_name

router = APIRouter()

NO_KEY = "Claude chat is off: ANTHROPIC_API_KEY is not set in the server's .env"
MAX_MESSAGE = 4000


def _settings(request: Request):
    settings = getattr(request.app.state, "assistant_settings", None)
    if settings is None:  # app started without its lifespan (should not happen in practice)
        raise ApiError(503, "assistant_unavailable", NO_KEY)
    return settings


def _require_chat(request: Request):
    if not _settings(request).chat_available:
        raise ApiError(503, "assistant_unavailable", NO_KEY)
    return request.app.state.assistant_chat


def _object(body: Any, allowed: set[str], required: set[str]) -> dict:
    if not isinstance(body, dict):
        raise ApiError(400, "bad_request", "body must be a JSON object")
    extra = sorted(set(body) - allowed)
    if extra:
        raise ApiError(400, "bad_request", f"unknown fields: {', '.join(extra)}")
    missing = sorted(required - set(body))
    if missing:
        raise ApiError(400, "bad_request", f"missing fields: {', '.join(missing)}")
    return body


def _user(body: dict) -> str:
    name = clean_name(body.get("user"))
    if name is None:
        raise ApiError(400, "bad_request",
                       "user must be 1-40 letters, digits, spaces, '.', '_' or '-'")
    return name


def _session_id(value: Any, required: bool) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise ApiError(400, "bad_request", "session_id must be a string")
    return value


@router.get("/assistant/status")
def status(request: Request):
    s = _settings(request)
    return {"chat_available": s.chat_available, "model": s.model,
            "max_tool_calls": s.max_tool_calls,
            "mcp": {"path": "/mcp/", "token_required": s.token_required}}


@router.post("/assistant/chat")
def chat(request: Request, body: Any = Body(None), conn: sqlite3.Connection = Depends(get_conn)):
    manager = _require_chat(request)
    body = _object(body, {"session_id", "project_id", "user", "message"},
                   {"project_id", "user", "message"})
    session_id = _session_id(body.get("session_id"), required=False)
    pid = body["project_id"]
    if not isinstance(pid, int) or isinstance(pid, bool):
        raise ApiError(400, "bad_request", "project_id must be an integer")
    user = _user(body)
    message = body["message"]
    if not isinstance(message, str) or not 1 <= len(message) <= MAX_MESSAGE:
        raise ApiError(400, "bad_request", f"message must be 1-{MAX_MESSAGE} characters")
    project = require_project(conn, pid)
    conn.close()  # the loop opens its own connections per tool call
    try:
        return manager.send(session_id, {"id": project["id"], "name": project["name"]},
                            user, message)
    except ChatError as e:
        raise ApiError(502, "assistant_error", str(e))


@router.post("/assistant/chat/confirm")
def confirm(request: Request, body: Any = Body(None)):
    manager = _require_chat(request)
    body = _object(body, {"session_id", "confirm_id", "decision", "user"},
                   {"session_id", "confirm_id", "decision", "user"})
    session_id = _session_id(body["session_id"], required=True)
    confirm_id = body["confirm_id"]
    if not isinstance(confirm_id, str) or not confirm_id:
        raise ApiError(400, "bad_request", "confirm_id must be a string")
    decision = body["decision"]
    if decision not in ("confirm", "cancel"):
        raise ApiError(400, "bad_request", "decision must be confirm or cancel")
    user = _user(body)
    return manager.confirm(session_id, confirm_id, decision, user)


@router.delete("/assistant/chat/{session_id}", status_code=204)
def forget(session_id: str, request: Request):
    manager = getattr(request.app.state, "assistant_chat", None)
    if manager is None:
        raise ApiError(404, "not_found", f"chat session {session_id} not found")
    manager.forget(session_id)
    return Response(status_code=204)


@router.get("/assistant/downloads/{token}")
def download(token: str, request: Request):
    found = downloads.find(request.app.state.data_dir, token)
    if found is None:
        raise ApiError(404, "not_found", "download not found or expired (links last 24 hours)")
    path, file_name = found
    return FileResponse(path, media_type="application/zip", filename=file_name)
