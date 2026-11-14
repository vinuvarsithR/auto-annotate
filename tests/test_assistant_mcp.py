"""#36 MCP server: tools/list and tools/call through the SDK's in-process client (mcp.Client),
and the HTTP rules at /mcp/ (Streamable HTTP, token 401, loopback-only 403, X-Autolabel-User)."""

import json

import anyio
import pytest
from mcp import Client

from app.assistant.mcp_server import build_server
from tests.test_assistant_helpers import (
    SAMPLES, TOOLS_FIXTURE, hermetic_env, load_fixture_data, make_app, open_client, rows, same_keys,
)

TOKEN = "test-mcp-token-123"
JSON_HEADERS = {"accept": "application/json, text/event-stream",
                "content-type": "application/json"}
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "test", "version": "1"}}}


def _rpc(method, params, rid=2):
    return {"jsonrpc": "2.0", "id": rid, "method": method, "params": params}


@pytest.fixture
def app_no_token(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path, token="")
    app = make_app(tmp_path)
    with open_client(app) as c:
        load_fixture_data(app)
        yield app, c


@pytest.fixture
def app_token(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path, token=TOKEN)
    app = make_app(tmp_path)
    with open_client(app, client_host="192.168.1.50", host="192.168.1.10:8000") as c:
        load_fixture_data(app)
        yield app, c


# ------------------------------------------------------------------ in-process (mcp.Client)

def _in_process(app, fn):
    async def main():
        async with Client(build_server(app.state)) as client:
            return await fn(client)
    return anyio.run(main)


def test_tools_list_equals_the_fixture(app_no_token):
    app, _ = app_no_token
    tools = _in_process(app, lambda c: c.list_tools()).tools
    fixture = TOOLS_FIXTURE["tools"]
    assert [t.name for t in tools] == [t["name"] for t in fixture]
    for got, want in zip(tools, fixture):
        assert got.description == want["description"]
        assert got.input_schema == want["input_schema"]  # confirm kept
        a = got.annotations
        if want["kind"] == "read":
            assert a.read_only_hint is True and a.destructive_hint is None
        elif want["name"] in ("delete_annotation", "push_to_roboflow"):
            assert a.destructive_hint is True and a.read_only_hint is False
        else:
            assert a.read_only_hint is False and a.destructive_hint is False


def test_call_returns_the_envelope_as_structured_content(app_no_token):
    app, _ = app_no_token
    result = _in_process(app, lambda c: c.call_tool("project_summary", {"project_id": 1}))
    want = TOOLS_FIXTURE["examples"]["project_summary"][0]["output"]
    assert result.structured_content == want
    assert result.is_error is False
    assert [c.type for c in result.content] == ["text"]
    assert json.loads(result.content[0].text) == want


def test_error_envelope_is_an_error_confirmation_is_not(app_no_token):
    app, _ = app_no_token
    err = _in_process(app, lambda c: c.call_tool("project_summary", {"project_id": 99}))
    assert err.is_error is True and err.structured_content["error"]["code"] == "not_found"
    # in-process has no HTTP request, so no X-Autolabel-User: writes are refused
    write = _in_process(app, lambda c: c.call_tool("delete_annotation", {"annotation_id": 106}))
    assert write.is_error is True and "X-Autolabel-User" in write.structured_content["error"]["message"]


def test_each_app_gets_its_own_server(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path)
    for i in range(2):  # a session manager runs only once: a second app must still start
        app = make_app(tmp_path / str(i))
        with open_client(app) as c:
            assert c.post("/mcp/", headers=JSON_HEADERS, json=INIT).status_code == 200


# ------------------------------------------------------------------ HTTP, no token

def test_loopback_without_token_works_over_http(app_no_token):
    app, c = app_no_token
    r = c.post("/mcp/", headers=JSON_HEADERS, json=INIT)
    assert r.status_code == 200 and r.json()["result"]["serverInfo"]["name"] == "autolabel"
    h = {**JSON_HEADERS, "mcp-protocol-version": "2025-06-18"}
    r = c.post("/mcp/", headers=h, json=_rpc("tools/list", {}))
    assert [t["name"] for t in r.json()["result"]["tools"]] == [
        t["name"] for t in TOOLS_FIXTURE["tools"]]
    r = c.post("/mcp/", headers={**h, "X-Autolabel-User": "dhyan"},
               json=_rpc("tools/call", {"name": "set_annotation_status",
                                        "arguments": {"annotation_ids": [102],
                                                      "action": "approve"}}))
    result = r.json()["result"]
    assert result["isError"] is False and result["structuredContent"]["ok"] is True
    assert rows(app, "SELECT reviewer FROM annotations WHERE id=102") == [
        {"reviewer": "claude:dhyan"}]
    assert rows(app, "SELECT channel, actor FROM assistant_actions") == [
        {"channel": "mcp", "actor": "claude:dhyan"}]


@pytest.mark.parametrize("name", [None, "", "a" * 41, "dhyan/../x", "<script>"])
def test_http_write_without_a_good_name_is_refused(app_no_token, name):
    _, c = app_no_token
    h = {**JSON_HEADERS, "mcp-protocol-version": "2025-06-18"}
    if name is not None:
        h["X-Autolabel-User"] = name
    r = c.post("/mcp/", headers=h, json=_rpc("tools/call", {
        "name": "add_class", "arguments": {"project_id": 1, "name": "hook"}}))
    result = r.json()["result"]
    assert result["isError"] is True
    assert result["structuredContent"]["error"]["code"] == "bad_request"
    r = c.post("/mcp/", headers=h, json=_rpc("tools/call", {
        "name": "list_projects", "arguments": {}}))
    assert r.json()["result"]["isError"] is False


@pytest.mark.parametrize("client_host", ["192.168.1.50", "10.0.0.2", "testclient"])
def test_no_token_rejects_non_loopback_clients(tmp_path, monkeypatch, client_host):
    hermetic_env(monkeypatch, tmp_path, token="")
    app = make_app(tmp_path)
    with open_client(app, client_host=client_host) as c:
        r = c.post("/mcp/", headers=JSON_HEADERS, json=INIT)
        assert r.status_code == 403
        assert r.json() == SAMPLES["error example (403 forbidden, /mcp/ no token set, not loopback)"]
        assert c.get("/api/projects").status_code == 200  # /api is unchanged


def test_ipv6_loopback_is_allowed(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path, token="")
    app = make_app(tmp_path)
    with open_client(app, client_host="::1", host="[::1]:8000") as c:
        assert c.post("/mcp/", headers=JSON_HEADERS, json=INIT).status_code == 200


def _app_with_static_ui(tmp_path):
    """Like make_app but with the real static UI mounted at /, as in production: a bare /mcp
    used to fall through to it and get 405."""
    from app.main import STATIC_DIR, create_app
    return create_app(tmp_path / "data", start_worker=False, static_dir=STATIC_DIR)


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_both_paths_list_tools_from_loopback(tmp_path, monkeypatch, path):
    hermetic_env(monkeypatch, tmp_path, token="")
    with open_client(_app_with_static_ui(tmp_path)) as c:
        h = {**JSON_HEADERS, "mcp-protocol-version": "2025-06-18"}
        r = c.post(path, headers=h, json=_rpc("tools/list", {}), follow_redirects=False)
        assert r.status_code == 200
        assert [t["name"] for t in r.json()["result"]["tools"]] == [
            t["name"] for t in TOOLS_FIXTURE["tools"]]
        assert c.get("/").status_code == 200  # the static UI is still served


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_both_paths_refuse_a_lan_client_without_token(tmp_path, monkeypatch, path):
    hermetic_env(monkeypatch, tmp_path, token="")
    with open_client(_app_with_static_ui(tmp_path), client_host="192.168.1.50") as c:
        r = c.post(path, headers=JSON_HEADERS, json=_rpc("tools/list", {}),
                   follow_redirects=False)
        assert r.status_code == 403
        assert r.json() == SAMPLES["error example (403 forbidden, /mcp/ no token set, not loopback)"]


@pytest.mark.parametrize("path", ["/mcp", "/mcp/"])
def test_both_paths_need_the_bearer_when_token_set(tmp_path, monkeypatch, path):
    hermetic_env(monkeypatch, tmp_path, token=TOKEN)
    with open_client(_app_with_static_ui(tmp_path), client_host="192.168.1.50",
                     host="192.168.1.10:8000") as c:
        assert c.post(path, headers=JSON_HEADERS, json=INIT,
                      follow_redirects=False).status_code == 401
        h = {**JSON_HEADERS, "authorization": f"Bearer {TOKEN}"}
        assert c.post(path, headers=h, json=INIT, follow_redirects=False).status_code == 200


def test_no_token_dns_rebinding_protection_is_on(app_no_token):
    _, c = app_no_token
    r = c.post("/mcp/", headers={**JSON_HEADERS, "host": "evil.example:8000"}, json=INIT)
    assert r.status_code != 200  # the SDK rejects a foreign Host header


# ------------------------------------------------------------------ HTTP, token set

@pytest.mark.parametrize("auth", [None, "Bearer wrong", f"Basic {TOKEN}", "Bearer ", TOKEN,
                                  f"Bearer {TOKEN}x"])
def test_token_set_needs_the_right_bearer(app_token, auth):
    _, c = app_token
    h = dict(JSON_HEADERS)
    if auth is not None:
        h["authorization"] = auth
    r = c.post("/mcp/", headers=h, json=INIT)
    assert r.status_code == 401
    assert r.json() == SAMPLES["error example (401 unauthorized, /mcp/ bad token)"]
    assert TOKEN not in r.text


def test_token_set_lan_client_with_token_works(app_token):
    app, c = app_token
    h = {**JSON_HEADERS, "authorization": f"Bearer {TOKEN}"}
    assert c.post("/mcp/", headers=h, json=INIT).status_code == 200
    r = c.post("/mcp/", headers={**h, "mcp-protocol-version": "2025-06-18"},
               json=_rpc("tools/call", {"name": "project_summary",
                                        "arguments": {"project_id": 1}}))
    assert r.status_code == 200
    assert r.json()["result"]["structuredContent"] == \
        TOOLS_FIXTURE["examples"]["project_summary"][0]["output"]


def test_token_set_api_is_unchanged(app_token):
    _, c = app_token
    assert c.get("/api/projects").status_code == 200


def test_error_bodies_match_fixture_keys(app_token):
    _, c = app_token
    same_keys(c.post("/mcp/", headers=JSON_HEADERS, json=INIT).json(),
              SAMPLES["error example (401 unauthorized, /mcp/ bad token)"])


# ------------------------------------------------------------------ #31 status

def test_status_with_key_and_token(app_token):
    _, c = app_token
    body = c.get("/api/assistant/status").json()
    assert body == SAMPLES["GET /api/assistant/status"]


def test_status_without_key_or_token(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path, key="", token="")
    app = make_app(tmp_path)
    with open_client(app) as c:
        assert c.get("/api/assistant/status").json() == SAMPLES["GET /api/assistant/status (no key)"]


def test_status_reads_model_and_cap_from_env(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path, AUTOLABEL_CLAUDE_MODEL="claude-opus-5-5",
                 AUTOLABEL_CLAUDE_MAX_TOOL_CALLS="4")
    app = make_app(tmp_path)
    with open_client(app) as c:
        body = c.get("/api/assistant/status").json()
        assert (body["model"], body["max_tool_calls"]) == ("claude-opus-5-5", 4)


def test_env_file_is_read_and_environment_wins(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path, key="")
    env_file = tmp_path / "test.env"
    env_file.write_text("AUTOLABEL_CLAUDE_MODEL=claude-from-file\nAUTOLABEL_CLAUDE_EFFORT=high\n",
                        encoding="utf-8")
    import app.assistant.settings as settings
    monkeypatch.setattr(settings, "ENV_FILE", env_file)
    monkeypatch.setenv("AUTOLABEL_CLAUDE_EFFORT", "medium")
    s = settings.load_settings()
    assert (s.model, s.effort) == ("claude-from-file", "medium")
    monkeypatch.delenv("AUTOLABEL_CLAUDE_MODEL")  # load_dotenv put it in os.environ
