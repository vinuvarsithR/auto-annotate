"""
mcp_server.py - MCP server #36 (docs/CONTRACT.md "MCP server").

* mcp 2.x low-level Server ("autolabel") with tools/list + tools/call over the tool layer.
* Streamable HTTP, stateless, JSON responses, mounted at /mcp, so the URL is http://HOST:PORT/mcp/.
* create_app builds a NEW Server per app and runs its session manager inside the app's lifespan.
* McpGate (ASGI, /mcp only) checks access before the SDK sees the request:
    AUTOLABEL_MCP_TOKEN set   -> "Authorization: Bearer <token>" (hmac.compare_digest), else 401;
                                 the SDK's DNS-rebinding protection is off (LAN host names vary)
    token empty / unset       -> loopback clients only (127.0.0.1, ::1), else 403; the SDK's
                                 DNS-rebinding protection on for 127.0.0.1 / localhost / [::1]
* Who: the X-Autolabel-User header (name rule in tools.clean_name). No or bad name -> read tools
  work, write tools return error bad_request.
"""

from __future__ import annotations

import hmac
import json
from typing import Any

import anyio
from starlette.responses import JSONResponse

from app.api.errors import error_body
from app.assistant.tools import TOOLS, ToolContext, call_tool, clean_name

DESTRUCTIVE = ("delete_annotation", "push_to_roboflow")
USER_HEADER = "x-autolabel-user"
LOOPBACK = ("127.0.0.1", "::1", "::ffff:127.0.0.1")
UNAUTHORIZED = "missing or wrong bearer token"
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
        user = clean_name(request.headers.get(USER_HEADER)) if request is not None else None
        envelope = await anyio.to_thread.run_sync(
            call_tool, ToolContext(state, "mcp", user), params.name, params.arguments)
        return envelope_result(envelope)

    return Server("autolabel", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def build_http_app(server, token_required: bool):
    """The SDK's Streamable HTTP app for `server`, to be wrapped by McpGate at /mcp."""
    from mcp.server.transport_security import TransportSecuritySettings

    if token_required:
        security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    else:
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=["127.0.0.1:*", "localhost:*", "[::1]:*"],
            allowed_origins=["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"])
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


class McpGate:
    """ASGI app mounted at /mcp: access check, then the SDK app built in the lifespan."""

    def __init__(self, state: Any):
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return
        settings = getattr(self.state, "assistant_settings", None)
        inner = getattr(self.state, "mcp_http_app", None)
        if settings is None or inner is None:
            response = JSONResponse(error_body("assistant_unavailable", "MCP is not running"),
                                    status_code=503)
            return await response(scope, receive, send)
        if settings.token_required:
            given = _bearer(scope).encode()
            if not hmac.compare_digest(given, settings.mcp_token.encode()):
                response = JSONResponse(error_body("unauthorized", UNAUTHORIZED), status_code=401,
                                        headers={"WWW-Authenticate": "Bearer"})
                return await response(scope, receive, send)
        else:
            client = scope.get("client")
            if not client or client[0] not in LOOPBACK:
                response = JSONResponse(error_body("forbidden", FORBIDDEN), status_code=403)
                return await response(scope, receive, send)
        await inner(scope, receive, send)
