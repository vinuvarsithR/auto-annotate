"""Neither secret (ANTHROPIC_API_KEY, AUTOLABEL_MCP_TOKEN) may leave the server: not in any
response body or header, log record, assistant_actions row or jobs row. Fake values only."""

import json
import logging

import anthropic
import httpx2

from tests.test_assistant_chat import FakeClient, end_turn, tool_turn
from tests.test_assistant_helpers import (
    hermetic_env, load_fixture_data, make_app, open_client, rows,
)

KEY = "sk-ant-test-LEAKME"
TOKEN = "mcp-token-LEAKME-42"
SECRETS = (KEY, TOKEN)
RPC = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
INIT = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                   "clientInfo": {"name": "t", "version": "1"}}}


class LeakyError(Exception):
    """A non-HTTP SDK error whose text carries both secrets."""


def test_no_secret_leaves_the_server(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    hermetic_env(monkeypatch, tmp_path, key=KEY, token=TOKEN)
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    fake = FakeClient([
        end_turn("hello"),                                                   # chat ok
        tool_turn(("get_job", {"job_id": 999}),                              # tool error
                  ("delete_annotation", {"annotation_id": 106, "confirm": True})),
        end_turn("Press Confirm"),
    ])
    app = make_app(tmp_path, anthropic_client=fake)
    responses = []
    with open_client(app, client_host="192.168.1.20", host="192.168.1.10:8000") as c:
        load_fixture_data(app)

        def keep(r):
            responses.append(r)
            return r

        status = keep(c.get("/api/assistant/status")).json()
        assert status["chat_available"] and status["mcp"]["token_required"]

        chat = {"project_id": 1, "user": "dhyan", "message": "hi"}
        sid = keep(c.post("/api/assistant/chat", json=chat)).json()["session_id"]
        body = keep(c.post("/api/assistant/chat", json={**chat, "session_id": sid})).json()
        assert [t["outcome"] for t in body["tool_calls"]] == ["error", "needs_confirmation"]
        cid = body["pending"][0]["confirm_id"]
        assert keep(c.post("/api/assistant/chat/confirm", json={
            "session_id": sid, "confirm_id": cid, "decision": "confirm", "user": "dhyan"})
        ).json()["result"]["ok"] is True

        # Claude API errors whose text contains the secrets
        fake.raises = anthropic.APIStatusError(
            f"invalid x-api-key {KEY} (token {TOKEN})",
            response=httpx2.Response(401, request=request), body=None)
        r = keep(c.post("/api/assistant/chat", json=chat))
        assert r.status_code == 502 and "[redacted]" in r.json()["error"]["message"]
        fake.raises = LeakyError(f"boom {KEY} {TOKEN}")
        assert keep(c.post("/api/assistant/chat", json=chat)).status_code == 502

        # MCP: wrong token, right token, a write and a read
        assert keep(c.post("/mcp/", headers={**RPC, "authorization": "Bearer nope"},
                           json=INIT)).status_code == 403  # wrong token: 403 since teammates
        auth = {**RPC, "authorization": f"Bearer {TOKEN}", "mcp-protocol-version": "2025-06-18",
                "X-Autolabel-User": "dhyan"}
        assert keep(c.post("/mcp/", headers=auth, json=INIT)).status_code == 200
        for name, args in (("set_annotation_status", {"annotation_ids": [102],
                                                      "action": "approve"}),
                           ("project_summary", {"project_id": 1}),
                           ("delete_annotation", {"annotation_id": 999, "confirm": True})):
            r = keep(c.post("/mcp/", headers=auth, json={
                "jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": name, "arguments": args}}))
            assert r.status_code == 200

    for r in responses:
        text = r.text + " " + " ".join(f"{k}: {v}" for k, v in r.headers.items())
        for s in SECRETS:
            assert s not in text, f"secret in response to {r.request.url}"
    log_text = caplog.text + " ".join(str(rec.args) for rec in caplog.records)
    assert "Claude API error 401: invalid x-api-key [redacted]" in log_text  # the log path ran
    for s in SECRETS:
        assert s not in log_text
    db_text = repr(rows(app, "SELECT * FROM assistant_actions")) + repr(rows(app, "SELECT * FROM jobs"))
    assert "claude:dhyan" in db_text
    for s in SECRETS:
        assert s not in db_text
    # the history Claude sees never carries them either
    for s in SECRETS:
        assert s not in repr(app.state.assistant_chat._sessions)


def test_settings_repr_hides_secrets(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path, key=KEY, token=TOKEN)
    from app.assistant.settings import load_settings
    s = load_settings()
    assert s.chat_available and s.token_required
    assert KEY not in repr(s) and TOKEN not in repr(s)


def test_client_is_built_with_the_key_only(tmp_path, monkeypatch):
    """With no fake client, the chat builds anthropic.Anthropic(api_key=KEY) (nothing else)."""
    hermetic_env(monkeypatch, tmp_path, key=KEY)
    seen = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", Recorder)
    app = make_app(tmp_path)
    with open_client(app):
        client = app.state.assistant_chat.client
    assert isinstance(client, Recorder) and set(seen) == {"api_key"} and seen["api_key"] == KEY


def test_a_teammate_token_never_leaves_the_server(tmp_path, monkeypatch, caplog):
    """The token add_teammate prints: not in any response, log record or assistant_actions row
    (neither when it works nor after it was revoked)."""
    from app.assistant import users as users_mod

    caplog.set_level(logging.DEBUG)
    users_file = tmp_path / "mcp_users.json"
    hermetic_env(monkeypatch, tmp_path, key=KEY, token="", AUTOLABEL_MCP_USERS_FILE=str(users_file))
    mate = users_mod.add_user(users_file, "mate@example.com")
    app = make_app(tmp_path)
    texts = []
    with open_client(app, client_host="192.168.1.20", host="192.168.1.10:8000") as c:
        load_fixture_data(app)
        auth = {**RPC, "authorization": f"Bearer {mate}", "mcp-protocol-version": "2025-06-18",
                "X-Autolabel-User": "liar"}
        for r in (
            c.post("/mcp/", headers=auth, json=INIT),
            c.post("/mcp/", headers=auth, json={"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                                "params": {"name": "add_class", "arguments": {
                                                    "project_id": 1, "name": "hook"}}}),
            c.post("/mcp/", headers=auth, json={"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                                "params": {"name": "get_job",
                                                           "arguments": {"job_id": 999}}}),
            c.get("/api/assistant/status"),
        ):
            texts.append(r.text + str(dict(r.headers)))
        users_mod.remove_user(users_file, "mate@example.com")
        r = c.post("/mcp/", headers=auth, json=INIT)
        assert r.status_code == 403
        texts.append(r.text + str(dict(r.headers)))
    db_text = json.dumps(rows(app, "SELECT * FROM assistant_actions"))
    assert "claude:mate@example.com" in db_text and "liar" not in db_text
    for where, text in [("responses", " ".join(texts)), ("logs", caplog.text), ("db", db_text)]:
        assert mate not in text, where
