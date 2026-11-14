"""app/assistant/mcp_stdio.py: the Claude Desktop launcher forwards tools/list and tools/call
upstream with both headers, returns results unchanged, and turns an unreachable upstream into
an is_error result. Also one end-to-end run against the real app over HTTP (loopback)."""

import ast
import socket
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
import uvicorn
from mcp import Client, types

from app.assistant import mcp_stdio
from tests.test_assistant_helpers import (
    TOOLS_FIXTURE, hermetic_env, load_fixture_data, make_app, rows, swap,
)

TOKEN = "stdio-token-xyz"
FILE = Path(mcp_stdio.__file__)


class FakeUpstream:
    def __init__(self):
        self.calls = []
        self.tools = types.ListToolsResult(tools=[types.Tool(
            name="ping", description="d", input_schema={"type": "object"})])
        self.result = types.CallToolResult(
            content=[types.TextContent(type="text", text='{"ok": true, "data": 1}')],
            structured_content={"ok": True, "data": 1}, is_error=False)

    async def list_tools(self):
        self.calls.append(("list",))
        return self.tools

    async def call_tool(self, name, arguments):
        self.calls.append(("call", name, arguments))
        return self.result


def _fake_connector(upstream, seen_headers):
    @asynccontextmanager
    async def connect(headers):
        seen_headers.append(headers)
        yield upstream
    return connect


def _run(server, fn):
    async def main():
        async with Client(server) as c:
            return await fn(c)
    return anyio.run(main)


def test_only_mcp_httpx2_and_stdlib_are_imported():
    tree = ast.parse(FILE.read_text(encoding="utf-8"))
    top = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add(node.module.split(".")[0])
    assert top - set(sys.stdlib_module_names) == {"mcp", "httpx2"}


def test_headers():
    assert mcp_stdio.Upstream("u", token=TOKEN, user="dhyan").headers() == {
        "Authorization": f"Bearer {TOKEN}", "X-Autolabel-User": "dhyan"}
    assert mcp_stdio.Upstream("u", user="dhyan").headers() == {"X-Autolabel-User": "dhyan"}
    client = mcp_stdio.Upstream("u", token=TOKEN, user="dhyan").http_client()
    assert client.headers["authorization"] == f"Bearer {TOKEN}"
    assert client.headers["x-autolabel-user"] == "dhyan"
    anyio.run(client.aclose)


def test_upstream_from_env():
    up = mcp_stdio.upstream_from_env("http://h/mcp/", {"AUTOLABEL_MCP_TOKEN": f" {TOKEN} ",
                                                       "AUTOLABEL_USER": " dhyan "})
    assert up.headers() == {"Authorization": f"Bearer {TOKEN}", "X-Autolabel-User": "dhyan"}
    assert mcp_stdio.upstream_from_env("http://h/mcp/", {}).headers() == {}


def test_forwards_list_and_call_unchanged_with_both_headers():
    fake, seen = FakeUpstream(), []
    up = mcp_stdio.Upstream("http://h/mcp/", token=TOKEN, user="dhyan",
                            connect=_fake_connector(fake, seen))
    server = mcp_stdio.build_server(up)
    listed = _run(server, lambda c: c.list_tools())
    assert [t.name for t in listed.tools] == ["ping"]
    result = _run(server, lambda c: c.call_tool("ping", {"a": 1}))
    assert result.structured_content == {"ok": True, "data": 1} and result.is_error is False
    assert result.content[0].text == '{"ok": true, "data": 1}'
    # (the SDK client may list tools again around a call: only the forwarded call is exact)
    assert [c for c in fake.calls if c[0] == "call"] == [("call", "ping", {"a": 1})]
    assert fake.calls[0] == ("list",)
    assert seen and all(h == {"Authorization": f"Bearer {TOKEN}", "X-Autolabel-User": "dhyan"}
                        for h in seen)


def test_unreachable_upstream_is_a_one_line_error():
    @asynccontextmanager
    async def broken(headers):
        raise ConnectionRefusedError(f"refused (token {TOKEN})\nsecond line")
        yield  # pragma: no cover

    up = mcp_stdio.Upstream("http://h/mcp/", token=TOKEN, user="d", connect=broken)
    result = _run(mcp_stdio.build_server(up), lambda c: c.call_tool("ping", {}))
    assert result.is_error is True
    text = result.content[0].text
    assert "\n" not in text and "unreachable" in text and TOKEN not in text


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_real_http_closed_port_is_an_error():
    up = mcp_stdio.Upstream(f"http://127.0.0.1:{_free_port()}/mcp/", user="d")
    result = _run(mcp_stdio.build_server(up), lambda c: c.call_tool("list_projects", {}))
    assert result.is_error is True and "unreachable" in result.content[0].text


def test_main_needs_url():
    with pytest.raises(SystemExit):
        mcp_stdio.main([])


@pytest.fixture
def live_app(tmp_path, monkeypatch):
    """The real app served by uvicorn on a free loopback port (no token: loopback only)."""
    hermetic_env(monkeypatch, tmp_path, token="")
    app = make_app(tmp_path)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning",
                                           lifespan="on"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        assert time.time() < deadline, "uvicorn did not start"
        time.sleep(0.05)
    load_fixture_data(app)
    yield app, f"http://127.0.0.1:{port}/mcp/"
    server.should_exit = True
    thread.join(10)


def test_end_to_end_through_the_real_app(live_app):
    app, url = live_app
    server = mcp_stdio.build_server(mcp_stdio.Upstream(url, user="dhyan"))
    listed = _run(server, lambda c: c.list_tools())
    assert [t.name for t in listed.tools] == [t["name"] for t in TOOLS_FIXTURE["tools"]]
    result = _run(server, lambda c: c.call_tool("project_summary", {"project_id": 1}))
    base = url.removesuffix("/mcp/")  # web_url BASE = the Host the stdio bridge connects to
    assert result.structured_content == swap(
        TOOLS_FIXTURE["examples"]["project_summary"][0]["output"], "http://localhost:8000", base)
    result = _run(server, lambda c: c.call_tool("add_class", {"project_id": 1, "name": "hook"}))
    assert result.is_error is False and result.structured_content["ok"] is True
    assert rows(app, "SELECT channel, actor, tool FROM assistant_actions") == [
        {"channel": "mcp", "actor": "claude:dhyan", "tool": "add_class"}]


# ------------------------------------------------------------------ --token (2026-11-14)

def _main_upstream(monkeypatch, argv):
    """Run main(argv) with serve() replaced; returns the Upstream it would have served."""
    got = []

    async def fake_serve(upstream):
        got.append(upstream)

    monkeypatch.setattr(mcp_stdio, "serve", fake_serve)
    assert mcp_stdio.main(argv) == 0
    return got[0]


def test_token_flag_beats_the_env(monkeypatch, capsys):
    monkeypatch.setenv("AUTOLABEL_MCP_TOKEN", "from-env")
    monkeypatch.delenv("AUTOLABEL_USER", raising=False)
    up = _main_upstream(monkeypatch, ["--url", "http://h/mcp/", "--token", TOKEN])
    assert up.headers() == {"Authorization": f"Bearer {TOKEN}"}
    err = capsys.readouterr().err
    assert "warning" not in err and TOKEN not in err  # with a token the name is optional


def test_env_token_without_flag_and_no_warning(monkeypatch, capsys):
    monkeypatch.setenv("AUTOLABEL_MCP_TOKEN", TOKEN)
    monkeypatch.delenv("AUTOLABEL_USER", raising=False)
    up = _main_upstream(monkeypatch, ["--url", "http://h/mcp/"])
    assert up.headers() == {"Authorization": f"Bearer {TOKEN}"}
    assert "warning" not in capsys.readouterr().err


def test_no_token_no_user_still_warns(monkeypatch, capsys):
    monkeypatch.delenv("AUTOLABEL_MCP_TOKEN", raising=False)
    monkeypatch.delenv("AUTOLABEL_USER", raising=False)
    _main_upstream(monkeypatch, ["--url", "http://h/mcp/"])
    assert "AUTOLABEL_USER" in capsys.readouterr().err


def test_upstream_from_env_token_argument():
    up = mcp_stdio.upstream_from_env("http://h/mcp/", {"AUTOLABEL_MCP_TOKEN": "env"}, token=" t ")
    assert up.headers() == {"Authorization": "Bearer t"}
    up = mcp_stdio.upstream_from_env("http://h/mcp/", {"AUTOLABEL_MCP_TOKEN": "env"}, token="")
    assert up.headers() == {"Authorization": "Bearer env"}
