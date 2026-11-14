"""
mcp_server.py - MCP server #36 (docs/CONTRACT.md "MCP server").

* mcp 2.x low-level Server ("autolabel") with tools/list + tools/call over the tool layer.
* Streamable HTTP, stateless, JSON responses, mounted at /mcp, so the URL is http://HOST:PORT/mcp/.
* create_app builds a NEW Server per app and runs its session manager inside the app's lifespan.
* McpGate (ASGI, /mcp only) checks access before the SDK sees the request (#36 + teammates):
    bearer = a teammate token (sha256 in the users file)  -> allowed; user = that email
    bearer = AUTOLABEL_MCP_TOKEN (shared, when set)       -> allowed; user from X-Autolabel-User
    bearer given, matches nothing                         -> 403 "unknown or revoked token"
    no bearer, shared token set                           -> 401
    no bearer, loopback client                            -> allowed if Host is 127.0.0.1 /
                                                             localhost / [::1] (any port), else 403
    no bearer, LAN client, >= 1 teammate                  -> 401
    no bearer, LAN client, no teammates                   -> 403 (local-only)
  The SDK's DNS-rebinding protection is off; the Host check above replaces it (per request, as
  teammates can appear without a restart).
* Who: the email of a teammate token (X-Autolabel-User ignored), else the X-Autolabel-User
  header (name rule in tools.clean_name). No or bad name -> read tools work, write tools return
  error bad_request. The gate passes the email in the ASGI scope key USER_SCOPE_KEY, which it
  always overwrites, so a client cannot set it.
"""

from __future__ import annotations

import hmac
import json
import re
from typing import Any

import anyio
from starlette.responses import JSONResponse

from app.api.errors import error_body
from app.assistant.tools import TOOLS, ToolContext, call_tool, clean_name

DESTRUCTIVE = ("delete_annotation", "push_to_roboflow")
USER_HEADER = "x-autolabel-user"
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
USER_SCOPE_KEY = "autolabel.user"
UNAUTHORIZED = "missing bearer token"
REVOKED = "unknown or revoked token"
BAD_HOST = "Host header must be 127.0.0.1, localhost or [::1] for MCP without a token"
_LOOPBACK_HOST = re.compile(r"^(127\.0\.0\.1|localhost|\[::1\])(:\d+)?$", re.I)
FORBIDDEN = "MCP is local-only: set AUTOLABEL_MCP_TOKEN in .env to allow LAN clients"


def tool_annotations(name: str, kind: str):
    import mcp_types as types

    if kind == "read":
        return types.ToolAnnotations(read_only_hint=True)
    return types.ToolAnnotations(read_only_hint=False, destructive_hint=name in DESTRUCTIVE)


def mcp_tools() -> list:
    """The tool list exactly as tools.json (confirm kept), with the annotations rule."""
    import mcp_types as types

    return [types.Tool(name=t["name"], description=t["description"],
                       input_schema=t["input_schema"],
                       annotations=tool_annotations(t["name"], t["kind"])) for t in TOOLS]


def envelope_result(envelope: dict):
    import mcp_types as types

    return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(envelope))],
                                structured_content=envelope, is_error="error" in envelope)


def build_server(state: Any):
    """A new low-level Server whose tools run against this app's state."""
    import mcp_types as types
    from mcp.server.lowlevel import Server

    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=mcp_tools())

    async def on_call_tool(ctx, params):
        request = getattr(ctx, "request", None)  # None for an in-process mcp.Client
        user = base_url = None
        if request is not None:
            user = request.scope.get(USER_SCOPE_KEY) or clean_name(request.headers.get(USER_HEADER))
            host = request.headers.get("host")
            base_url = f"http://{host}" if host else None  # web_url BASE (E2E gaps item 6)
        envelope = await anyio.to_thread.run_sync(
            call_tool, ToolContext(state, "mcp", user, base_url=base_url), params.name,
            params.arguments)
        return envelope_result(envelope)

    return Server("autolabel", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def build_http_app(server):
    """The SDK's Streamable HTTP app for `server`, to be wrapped by McpGate at /mcp. The SDK's
    DNS-rebinding protection is off: McpGate checks the Host of tokenless loopback requests."""
    from mcp.server.transport_security import TransportSecuritySettings

    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    return server.streamable_http_app(streamable_http_path="/", stateless_http=True,
                                      json_response=True, transport_security=security)


def _bearer(scope) -> str:
    for key, value in scope.get("headers") or []:
        if key.lower() == b"authorization":
            text = value.decode("latin-1")
            if text[:7].lower() == "bearer ":
                return text[7:].strip()
            return ""
    return ""


def _header(scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or []:
        if key.lower() == name:
            return value.decode("latin-1")
    return None


def _deny(status: int, code: str, message: str, **headers) -> JSONResponse:
    return JSONResponse(error_body(code, message), status_code=status, headers=headers or None)


class McpGate:
    """ASGI app mounted at /mcp: access check, then the SDK app built in the lifespan."""

    def __init__(self, state: Any):
        self.state = state

    def check(self, scope) -> tuple[JSONResponse | None, str | None]:
        """(refusal or None, teammate email or None)."""
        settings = self.state.assistant_settings
        users = getattr(self.state, "mcp_users", None)
        token = _bearer(scope)
        if token:
            email = users.match(token) if users is not None else None
            if email is not None:
                return None, email
            if settings.token_required and hmac.compare_digest(token.encode(),
                                                               settings.mcp_token.encode()):
                return None, None
            return _deny(403, "forbidden", REVOKED), None
        if settings.token_required:
            return _deny(401, "unauthorized", UNAUTHORIZED, **{"WWW-Authenticate": "Bearer"}), None
        client = scope.get("client")
        if client and client[0] in LOOPBACK:
            if not _LOOPBACK_HOST.match(_header(scope, b"host") or ""):
                return _deny(403, "forbidden", BAD_HOST), None
            return None, None
        if users is not None and users.count() >= 1:
            return _deny(401, "unauthorized", UNAUTHORIZED, **{"WWW-Authenticate": "Bearer"}), None
        return _deny(403, "forbidden", FORBIDDEN), None

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return
        settings = getattr(self.state, "assistant_settings", None)
        inner = getattr(self.state, "mcp_http_app", None)
        if settings is None or inner is None:
            response = JSONResponse(error_body("assistant_unavailable", "MCP is not running"),
                                    status_code=503)
            return await response(scope, receive, send)
        refusal, email = self.check(scope)
        if refusal is not None:
            return await refusal(scope, receive, send)
        scope[USER_SCOPE_KEY] = email  # always set: a client cannot smuggle its own
        await inner(scope, receive, send)


class McpNoSlash:
    """ASGI app for the exact path /mcp (no slash). The /mcp mount only matches /mcp/..., so
    without this a POST /mcp fell through to the static UI (405). No redirect (MCP clients may
    not follow a POST redirect): the scope is rewritten to what the mount gives /mcp/, then the
    same gate runs, so the token / loopback check applies to both paths."""

    def __init__(self, gate: McpGate):
        self.gate = gate

    async def __call__(self, scope, receive, send):
        root = scope.get("root_path", "")
        path = root + "/mcp/"
        scope = dict(scope, path=path, raw_path=path.encode(), root_path=root + "/mcp")
        await self.gate(scope, receive, send)
