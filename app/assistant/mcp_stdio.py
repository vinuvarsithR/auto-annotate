"""
mcp_stdio.py - stdio MCP launcher for Claude Desktop (docs/CONTRACT.md "MCP server", #36).

ONE standalone file: it imports only `mcp`, `httpx2` and the standard library, so a teammate can
copy just this file. It is an MCP server on stdin/stdout that forwards tools/list and tools/call
to the app's /mcp/ URL (Streamable HTTP) and returns the upstream results unchanged.

Needs Python >= 3.11 and `pip install mcp==2.3.0`.

    python mcp_stdio.py --url http://HOST:8000/mcp/ [--token TOKEN]

Environment (set in claude_desktop_config.json "env"):
    AUTOLABEL_MCP_TOKEN   your teammate token (or the shared token), sent as
                          "Authorization: Bearer ..." (never printed or logged). --token wins over
                          it, but prefer the env: other local users can see command-line arguments.
    AUTOLABEL_USER        sent as "X-Autolabel-User" (your name). Needed for write tools only
                          without a teammate token (the server then uses the token's email).

Upstream unreachable -> the tool call returns is_error with a one-line message.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Callable

import httpx2
from mcp import Client, types
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

# connect(headers) -> async context manager yielding an object with list_tools() / call_tool()
Connector = Callable[[dict], Any]


class Upstream:
    """Where tools/list and tools/call go: the app's /mcp/ URL, one short connection per call
    (the app's MCP server is stateless)."""

    def __init__(self, url: str, token: str = "", user: str = "",
                 connect: Connector | None = None):
        self.url = url
        self._token = token
        self.user = user
        self._connect = connect or self._http_connect

    @property
    def has_token(self) -> bool:
        return bool(self._token)

    def headers(self) -> dict:
        h = {}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        if self.user:
            h["X-Autolabel-User"] = self.user
        return h

    def http_client(self) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(headers=self.headers(), timeout=httpx2.Timeout(30.0, read=300.0))

    @asynccontextmanager
    async def _http_connect(self, headers: dict) -> AsyncIterator[Client]:
        async with self.http_client() as hc:
            async with Client(streamable_http_client(self.url, http_client=hc)) as client:
                yield client

    @asynccontextmanager
    async def session(self) -> AsyncIterator[Any]:
        async with self._connect(self.headers()) as s:
            yield s

    def one_line(self, exc: BaseException) -> str:
        while isinstance(exc, BaseExceptionGroup) and exc.exceptions:  # anyio task groups
            exc = exc.exceptions[0]
        text = f"{type(exc).__name__}: {exc}".strip().splitlines()[0][:300]
        if self._token:
            text = text.replace(self._token, "[redacted]")
        return f"autolabel server at {self.url} unreachable or failed: {text}"


def build_server(upstream: Upstream) -> Server:
    async def on_list_tools(ctx, params):
        async with upstream.session() as s:
            return await s.list_tools()

    async def on_call_tool(ctx, params):
        try:
            async with upstream.session() as s:
                return await s.call_tool(params.name, params.arguments or {})
        except Exception as exc:  # noqa: BLE001 - any upstream failure becomes is_error
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=upstream.one_line(exc))],
                is_error=True)

    return Server("autolabel", on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def upstream_from_env(url: str, env=None, token: str | None = None) -> Upstream:
    """`token` (from --token) wins over AUTOLABEL_MCP_TOKEN."""
    env = os.environ if env is None else env
    token = (token or "").strip() or (env.get("AUTOLABEL_MCP_TOKEN") or "").strip()
    return Upstream(url, token=token, user=(env.get("AUTOLABEL_USER") or "").strip())


async def serve(upstream: Upstream) -> None:
    server = build_server(upstream)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="stdio MCP launcher for the autolabel app")
    parser.add_argument("--url", required=True, help="the app's MCP URL, e.g. http://HOST:8000/mcp/")
    parser.add_argument("--token", default=None,
                        help="bearer token (wins over AUTOLABEL_MCP_TOKEN; the env is safer)")
    args = parser.parse_args(argv)
    upstream = upstream_from_env(args.url, token=args.token)
    if not upstream.user and not upstream.has_token:
        print("warning: AUTOLABEL_USER is not set; write tools will refuse", file=sys.stderr)
    asyncio.run(serve(upstream))
    return 0


if __name__ == "__main__":
    sys.exit(main())
