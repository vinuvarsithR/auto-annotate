"""#32 - #34 sidebar chat: every docs/fixtures/assistant/chat_fake.json scenario with a fake
Anthropic client, plus the loop rules (cap, stop reasons, pending, memory limits, errors)."""

import copy
import threading

import anthropic
import httpx2
import pytest

import app.assistant.chat as chat_mod
from app.assistant.chat import CHAT_TOOLS, LIMIT_MESSAGE, SYSTEM
from tests.test_assistant_helpers import (
    CHAT_FIXTURE, SAMPLES, TOOLS_FIXTURE, hermetic_env, load_fixture_data, make_app, open_client,
    rows, same_keys,
)

SCENARIOS = {s["name"]: s for s in CHAT_FIXTURE["scenarios"]}
FIXTURE_ENV_KEYS = ("ANTHROPIC_API_KEY", "AUTOLABEL_CLAUDE_MAX_TOOL_CALLS")


class FakeMessages:
    def __init__(self, owner, beta: bool):
        self.owner, self.beta = owner, beta

    def create(self, **kwargs):
        self.owner.requests.append({"beta": self.beta, **kwargs})
        if self.owner.raises is not None:
            raise self.owner.raises
        if not self.owner.turns:
            raise AssertionError("the fake client ran out of scripted turns")
        return copy.deepcopy(self.owner.turns.pop(0))


class FakeClient:
    """Stands in for anthropic.Anthropic: .messages.create and .beta.messages.create."""

    def __init__(self, turns=(), raises=None):
        self.turns = list(turns)
        self.raises = raises
        self.requests: list[dict] = []
        self.messages = FakeMessages(self, beta=False)
        self.beta = type("Beta", (), {})()
        self.beta.messages = FakeMessages(self, beta=True)


def api_status_error(status: int, message: str) -> anthropic.APIStatusError:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.APIStatusError(message, response=httpx2.Response(status, request=request),
                                    body=None)


def end_turn(text="ok", usage=(10, 2)):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}


def tool_turn(*uses, stop="tool_use"):
    return {"stop_reason": stop,
            "content": [{"type": "tool_use", "id": f"tu_{i}", "name": n, "input": a}
                        for i, (n, a) in enumerate(uses)],
            "usage": {"input_tokens": 5, "output_tokens": 5}}


@pytest.fixture
def make(tmp_path, monkeypatch):
    """make(fake, **env) -> (app, client) with the fixture data; closed at the end."""
    opened = []

    def _make(fake=None, key="sk-ant-test-fake", **env):
        hermetic_env(monkeypatch, tmp_path, key=key, **env)
        app = make_app(tmp_path, anthropic_client=fake or FakeClient())
        c = open_client(app)
        c.__enter__()
        opened.append(c)
        load_fixture_data(app)
        return app, c

    yield _make
    for c in opened:
        c.__exit__(None, None, None)


def history(app, session_id):
    return app.state.assistant_chat._sessions[session_id].history


def _types(message):
    content = message["content"]
    return [b["type"] for b in content] if isinstance(content, list) else ["text"]


# ------------------------------------------------------------------ chat_fake.json scenarios

@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario(make, name):
    sc = SCENARIOS[name]
    env = dict(sc.get("env", {}))
    key = env.pop("ANTHROPIC_API_KEY", "sk-ant-test-fake")
    raises = None
    if "fake_raises" in sc:
        fr = sc["fake_raises"]
        assert fr["type"] == "APIStatusError"
        raises = api_status_error(fr["status_code"], fr["message"])
    fake = FakeClient(sc.get("model_turns", []), raises=raises)
    app, c = make(fake, key=key, **env)

    r = c.post("/api/assistant/chat", json=sc["request"])
    assert r.status_code == sc.get("expect_status", 200), r.text
    body = r.json()
    want = sc["expect_response"]
    if "error" in want:
        assert body == want
    else:
        same_keys(body, want)
        assert {k: v for k, v in body.items() if k not in ("session_id", "pending")} == \
            {k: v for k, v in want.items() if k not in ("session_id", "pending")}
        assert [{k: v for k, v in p.items() if k != "confirm_id"} for p in body["pending"]] == \
            [{k: v for k, v in p.items() if k != "confirm_id"} for p in want["pending"]]
    if "must_not_contain_anywhere" in sc:
        secret = sc["must_not_contain_anywhere"]
        assert secret not in r.text and secret not in str(dict(r.headers))

    exp = sc.get("expect_requests", {})
    if "count" in exp:
        assert len(fake.requests) == exp["count"]
    for req in fake.requests:
        for k in ("model", "max_tokens", "output_config", "betas", "fallbacks"):
            if k in exp:
                assert req[k] == exp[k], k
        if "tool_names" in exp:
            assert [t["name"] for t in req["tools"]] == exp["tool_names"]
        if exp.get("no_confirm_property"):
            assert all("confirm" not in t["input_schema"]["properties"] for t in req["tools"])
    if "second_request_ends_with" in exp:
        tail = fake.requests[1]["messages"][-len(exp["second_request_ends_with"]):]
        assert [{"role": m["role"], "content_types": _types(m)} for m in tail] == \
            exp["second_request_ends_with"]
    if "first_user_message_starts_with" in exp:
        first = fake.requests[0]["messages"][0]
        assert first["role"] == "user"
        assert first["content"][0]["text"].startswith(exp["first_user_message_starts_with"])

    if "confirm_request" in sc:
        sid, cid = body["session_id"], body["pending"][0]["confirm_id"]
        assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id=106")[0]["n"] == 1
        req = {**sc["confirm_request"], "session_id": sid, "confirm_id": cid}
        r2 = c.post("/api/assistant/chat/confirm", json=req)
        assert r2.status_code == 200, r2.text
        assert r2.json() == {**sc["expect_confirm_response"], "confirm_id": cid}
        last = history(app, sid)[-1]
        assert last["role"] == "user" and last["content"][-1] == {
            "type": "text", "text": sc["expect_history_note"]}
        assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id=106")[0]["n"] == 0


def test_cap_gives_the_extra_tool_use_an_error_result(make):
    sc = SCENARIOS["tool-call cap"]
    fake = FakeClient(sc["model_turns"])
    app, c = make(fake, **sc["env"])
    body = c.post("/api/assistant/chat", json=sc["request"]).json()
    results = history(app, body["session_id"])[-1]["content"]
    assert [(r["tool_use_id"], r["is_error"]) for r in results] == [
        ("t1", False), ("t2", False), ("t3", True)]
    assert results[2]["content"] == LIMIT_MESSAGE


def test_refusal_removes_the_user_turn(make):
    fake = FakeClient([end_turn("hello"), SCENARIOS["refusal"]["model_turns"][0]])
    app, c = make(fake)
    sid = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "hi"}
                 ).json()["session_id"]
    before = copy.deepcopy(history(app, sid))
    body = c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "d",
                                               "message": "..."}).json()
    assert body["stopped"] == "refusal"
    assert history(app, sid) == before


def test_risky_confirm_claude_sends_is_dropped(make):
    sc = SCENARIOS["risky tool becomes a confirm button; confirm runs it"]
    app, c = make(FakeClient(sc["model_turns"]))
    c.post("/api/assistant/chat", json=sc["request"])
    assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id=106")[0]["n"] == 1
    assert rows(app, "SELECT arguments, outcome, confirmed FROM assistant_actions") == [
        {"arguments": '{"annotation_id": 106}', "outcome": "needs_confirmation", "confirmed": 0}]


# ------------------------------------------------------------------ requests

def test_fixed_prefix_is_identical_on_every_request(make):
    fake = FakeClient([tool_turn(("list_projects", {})), end_turn(), end_turn()])
    app, c = make(fake)
    sid = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}
                 ).json()["session_id"]
    c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "d",
                                        "message": "b"})
    first = fake.requests[0]
    for req in fake.requests:
        assert req["system"] == SYSTEM and req["tools"] == CHAT_TOOLS == first["tools"]
        assert req["cache_control"] == {"type": "ephemeral"}
        assert "thinking" not in req and "tool_choice" not in req and "stream" not in req
        assert req["beta"] is True
    # the history is append-only: every request starts with the previous one's messages
    for a, b in zip(fake.requests, fake.requests[1:]):
        assert b["messages"][:len(a["messages"]) - 1] == a["messages"][:-1]


def test_chat_tools_are_the_fixture_without_confirm():
    for got, want in zip(CHAT_TOOLS, TOOLS_FIXTURE["tools"]):
        schema = copy.deepcopy(want["input_schema"])
        schema["properties"].pop("confirm", None)
        assert got == {"name": want["name"], "description": want["description"],
                       "input_schema": schema}
    assert len(CHAT_TOOLS) == 17


@pytest.mark.parametrize("model,beta", [("claude-sonnet-5-5", True), ("claude-opus-5-5", True),
                                        ("claude-opus-5", True), ("claude-fable-5-1", True),
                                        ("claude-fable-5-2", True), ("claude-haiku-5-5", False),
                                        ("claude-sonnet-4-5", False)])
def test_fallbacks_only_for_listed_models(make, model, beta):
    fake = FakeClient([end_turn()])
    _, c = make(fake, AUTOLABEL_CLAUDE_MODEL=model, AUTOLABEL_CLAUDE_EFFORT="medium",
                AUTOLABEL_CLAUDE_MAX_TOKENS="999")
    c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"})
    req = fake.requests[0]
    assert req["beta"] is beta and req["model"] == model
    assert req["max_tokens"] == 999 and req["output_config"] == {"effort": "medium"}
    if beta:
        assert req["betas"] == ["server-side-fallback-2026-07-01"]
        assert req["fallbacks"] == "default"
    else:
        assert "betas" not in req and "fallbacks" not in req


# ------------------------------------------------------------------ stop reasons

def test_pause_turn_calls_again(make):
    fake = FakeClient([{"stop_reason": "pause_turn", "content": [{"type": "text", "text": "…"}],
                        "usage": {"input_tokens": 3, "output_tokens": 1}}, end_turn("done")])
    _, c = make(fake)
    body = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}).json()
    assert (body["reply"], body["stopped"], len(fake.requests)) == ("done", "end_turn", 2)
    assert body["usage"] == {"input_tokens": 13, "output_tokens": 3}


def test_max_tokens_with_a_dangling_tool_use(make):
    fake = FakeClient([tool_turn(("list_projects", {}), stop="max_tokens")])
    app, c = make(fake)
    body = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}).json()
    assert body["stopped"] == "max_tokens" and body["tool_calls"] == []
    last = history(app, body["session_id"])[-1]
    assert last["role"] == "user" and last["content"][0]["type"] == "tool_result"
    assert last["content"][0]["is_error"] is True


def test_reply_joins_text_blocks_of_the_last_response(make):
    fake = FakeClient([{"stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
                        "content": [{"type": "text", "text": "one"},
                                    {"type": "thinking", "thinking": "", "signature": "s"},
                                    {"type": "text", "text": "two"}]}])
    _, c = make(fake)
    body = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}).json()
    assert body["reply"] == "one\n\ntwo"


def test_tool_error_goes_to_claude_not_http(make):
    fake = FakeClient([tool_turn(("get_job", {"job_id": 999})), end_turn("no such job")])
    app, c = make(fake)
    r = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"})
    assert r.status_code == 200
    assert r.json()["tool_calls"][0]["outcome"] == "error"
    result = fake.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True and '"not_found"' in result["content"]


def test_write_tool_in_chat_uses_the_body_user(make):
    fake = FakeClient([tool_turn(("set_annotation_status",
                                  {"annotation_ids": [102], "action": "reject"})), end_turn()])
    app, c = make(fake)
    c.post("/api/assistant/chat", json={"project_id": 1, "user": "ana", "message": "a"})
    assert rows(app, "SELECT reviewer FROM annotations WHERE id=102") == [{"reviewer": "claude:ana"}]
    assert rows(app, "SELECT channel, actor FROM assistant_actions") == [
        {"channel": "chat", "actor": "claude:ana"}]


# ------------------------------------------------------------------ #33 confirm

def _risky(make):
    fake = FakeClient([tool_turn(("delete_annotation", {"annotation_id": 106})),
                       end_turn("Press Confirm"), end_turn("ok")])
    app, c = make(fake)
    body = c.post("/api/assistant/chat", json={"project_id": 1, "user": "dhyan",
                                               "message": "delete 106"}).json()
    return app, c, fake, body["session_id"], body["pending"][0]["confirm_id"]


def test_cancel_runs_nothing(make):
    app, c, _, sid, cid = _risky(make)
    r = c.post("/api/assistant/chat/confirm", json={"session_id": sid, "confirm_id": cid,
                                                    "decision": "cancel", "user": "dhyan"})
    assert r.json() == {"confirm_id": cid, "tool": "delete_annotation", "decision": "cancel",
                        "result": None}
    assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id=106")[0]["n"] == 1
    note = history(app, sid)[-1]["content"][-1]["text"]
    assert note.startswith('[dhyan pressed Cancel: delete_annotation {"annotation_id": 106}')


def test_confirm_twice_is_409_and_the_row_says_confirmed(make):
    app, c, _, sid, cid = _risky(make)
    req = {"session_id": sid, "confirm_id": cid, "decision": "confirm", "user": "dhyan"}
    assert c.post("/api/assistant/chat/confirm", json=req).status_code == 200
    r = c.post("/api/assistant/chat/confirm", json=req)
    assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
    assert rows(app, "SELECT outcome, confirmed, arguments FROM assistant_actions ORDER BY id")[-1] \
        == {"outcome": "ok", "confirmed": 1, "arguments": '{"annotation_id": 106}'}


def test_next_message_expires_pending(make):
    app, c, fake, sid, cid = _risky(make)
    c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "dhyan",
                                        "message": "never mind"})
    r = c.post("/api/assistant/chat/confirm", json={"session_id": sid, "confirm_id": cid,
                                                    "decision": "confirm", "user": "dhyan"})
    assert r.status_code == 409
    assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id=106")[0]["n"] == 1


def test_confirm_note_reaches_claude_on_the_next_message(make):
    app, c, fake, sid, cid = _risky(make)
    c.post("/api/assistant/chat/confirm", json={"session_id": sid, "confirm_id": cid,
                                                "decision": "confirm", "user": "dhyan"})
    c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "dhyan",
                                        "message": "thanks"})
    last_user = fake.requests[-1]["messages"][-1]
    texts = [b["text"] for b in last_user["content"] if b["type"] == "text"]
    assert texts[0] == '[dhyan pressed Confirm: delete_annotation {"annotation_id": 106} -> ok]'
    assert texts[1].endswith("thanks")


@pytest.mark.parametrize("body,status", [
    ({"confirm_id": "x", "decision": "confirm", "user": "d"}, 400),
    ({"session_id": "s", "confirm_id": "x", "decision": "maybe", "user": "d"}, 400),
    ({"session_id": "s", "confirm_id": "x", "decision": "confirm", "user": "d", "x": 1}, 400),
    ({"session_id": "s", "confirm_id": "x", "decision": "confirm", "user": "bad/name"}, 400),
    ({"session_id": "nope", "confirm_id": "x", "decision": "confirm", "user": "d"}, 404),
])
def test_confirm_errors(make, body, status):
    _, c = make()
    assert c.post("/api/assistant/chat/confirm", json=body).status_code == status


# ------------------------------------------------------------------ #32 body and memory rules

@pytest.mark.parametrize("body", [
    {"project_id": 1, "user": "d"},                                     # no message
    {"project_id": 1, "user": "d", "message": ""},
    {"project_id": 1, "user": "d", "message": "x" * 4001},
    {"project_id": 1, "user": "", "message": "a"},
    {"project_id": 1, "user": "x" * 41, "message": "a"},
    {"project_id": 1, "user": "a/b", "message": "a"},
    {"project_id": "1", "user": "d", "message": "a"},
    {"project_id": 1, "user": "d", "message": "a", "model": "x"},     # extra field
    {"project_id": 1, "user": "d", "message": "a", "session_id": 5},
    [1, 2],
])
def test_bad_chat_body_is_400(make, body):
    _, c = make()
    r = c.post("/api/assistant/chat", json=body)
    assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"


def test_message_of_4000_chars_and_trimmed_user_are_fine(make):
    fake = FakeClient([end_turn()])
    _, c = make(fake)
    r = c.post("/api/assistant/chat", json={"project_id": 1, "user": "  d.k_1-x ", "message": "x" * 4000})
    assert r.status_code == 200
    assert fake.requests[0]["messages"][0]["content"][0]["text"].startswith(
        '[project 1 "eot_crane"; user d.k_1-x]\n')


def test_error_order_no_key_first(make):
    _, c = make(key="")
    r = c.post("/api/assistant/chat", json={"bad": 1})
    assert r.status_code == 503 and r.json() == SAMPLES[
        "error example (503 assistant_unavailable, #32 no key)"]
    r = c.post("/api/assistant/chat/confirm", json={})
    assert r.status_code == 503


def test_unknown_project_then_unknown_session(make):
    _, c = make()
    r = c.post("/api/assistant/chat", json={"session_id": "nope", "project_id": 99, "user": "d",
                                            "message": "a"})
    assert r.status_code == 404 and "project 99" in r.json()["error"]["message"]
    r = c.post("/api/assistant/chat", json={"session_id": "nope", "project_id": 1, "user": "d",
                                            "message": "a"})
    assert r.status_code == 404 and "memory" in r.json()["error"]["message"]


def test_delete_session(make):
    fake = FakeClient([end_turn()])
    _, c = make(fake)
    sid = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}
                 ).json()["session_id"]
    r = c.delete(f"/api/assistant/chat/{sid}")
    assert r.status_code == 204 and r.content == b""
    assert c.delete(f"/api/assistant/chat/{sid}").status_code == 404
    r = c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "d",
                                            "message": "a"})
    assert r.status_code == 404


def test_40_messages_then_409(make, monkeypatch):
    monkeypatch.setattr(chat_mod, "MAX_USER_MESSAGES", 3)
    fake = FakeClient([end_turn()] * 5)
    _, c = make(fake)
    sid = None
    for _ in range(3):
        body = c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "d",
                                                   "message": "a"}).json()
        sid = body["session_id"]
    r = c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "d",
                                            "message": "a"})
    assert r.status_code == 409 and "new chat" in r.json()["error"]["message"]
    assert chat_mod.MAX_USER_MESSAGES == 3 and chat_mod.MAX_SESSIONS == 200


def test_oldest_session_dropped(make, monkeypatch):
    monkeypatch.setattr(chat_mod, "MAX_SESSIONS", 2)
    fake = FakeClient([end_turn()] * 4)
    _, c = make(fake)
    ids = [c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}
                  ).json()["session_id"] for _ in range(3)]
    r = c.post("/api/assistant/chat", json={"session_id": ids[0], "project_id": 1, "user": "d",
                                            "message": "a"})
    assert r.status_code == 404
    r = c.post("/api/assistant/chat", json={"session_id": ids[2], "project_id": 1, "user": "d",
                                            "message": "a"})
    assert r.status_code == 200


def test_busy_session_is_409(make):
    fake = FakeClient([end_turn()])
    app, c = make(fake)
    sid = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}
                 ).json()["session_id"]
    lock = app.state.assistant_chat._sessions[sid].lock
    assert lock.acquire(blocking=False)
    try:
        r = c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "d",
                                                "message": "b"})
        assert r.status_code == 409
    finally:
        lock.release()


def test_two_messages_at_once_one_gets_409(make):
    started, release = threading.Event(), threading.Event()

    class SlowClient(FakeClient):
        pass

    fake = SlowClient([end_turn(), end_turn()])
    orig = fake.beta.messages.create

    def slow(**kw):
        started.set()
        release.wait(5)
        return orig(**kw)

    fake.beta.messages.create = slow
    app, c = make(fake)
    fake_first = FakeClient([end_turn()])  # first message on a normal path to get a session
    app.state.assistant_chat._client = fake_first
    sid = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}
                 ).json()["session_id"]
    app.state.assistant_chat._client = fake
    out = {}

    def first():
        out["a"] = c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1,
                                                       "user": "d", "message": "b"}).status_code

    t = threading.Thread(target=first)
    t.start()
    assert started.wait(5)
    r = c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "d",
                                            "message": "c"})
    release.set()
    t.join(5)
    assert r.status_code == 409 and out["a"] == 200


# ------------------------------------------------------------------ Claude API errors

def test_api_error_keeps_history_clean_for_a_retry(make):
    fake = FakeClient([end_turn("first")])
    app, c = make(fake)
    sid = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"}
                 ).json()["session_id"]
    before = copy.deepcopy(history(app, sid))
    fake.raises = api_status_error(429, "rate limited")
    r = c.post("/api/assistant/chat", json={"session_id": sid, "project_id": 1, "user": "d",
                                            "message": "b"})
    assert r.status_code == 502
    assert r.json() == SAMPLES["error example (502 assistant_error, #32 Claude API error)"]
    assert history(app, sid) == before


def test_connection_error_says_unreachable(make):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    _, c = make(FakeClient(raises=anthropic.APIConnectionError(request=request)))
    r = c.post("/api/assistant/chat", json={"project_id": 1, "user": "d", "message": "a"})
    assert r.status_code == 502
    assert r.json() == {"error": {"code": "assistant_error", "message": "Claude API unreachable"}}
