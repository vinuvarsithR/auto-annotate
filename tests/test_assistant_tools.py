"""Claude connector tool layer (app/assistant/tools.py): every docs/fixtures/assistant/tools.json
example against a DB built from api_samples.json, plus the contract's rules (input check, order,
confirmation thresholds, all-or-nothing approve, assistant_actions rows, reviewer, names, size)."""

import json
import zipfile
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest

import app.api.export as export_api
from app.assistant import downloads
from app.assistant.tools import READ_TOOLS, WRITE_TOOLS, ToolContext, call_tool
from tests.test_assistant_helpers import (
    TOOLS_FIXTURE, add_images, connect, fake_inspect, fake_pt,
    hermetic_env, load_fixture_data, make_app, metrics_file, open_client, rows, same_keys, swap,
)

FAKE_SETTINGS = SimpleNamespace(workspace="w", project="p", api_key="rf-fake")
OWL = {"detectors": ("fake", "owlv2"), "modes": {"fake": ["text", "examples"],
                                                 "owlv2": ["text", "examples"]}}


def _delete_job_12(app, monkeypatch):
    conn = connect(app)
    with conn:
        conn.execute("DELETE FROM jobs WHERE id=12")
    conn.close()


def _boxes_201_212(app, monkeypatch):
    conn = connect(app)
    with conn:
        for aid in range(201, 213):
            conn.execute("INSERT INTO annotations (id, image_id, class_id, x, y, w, h, score, "
                         "source, status, job_id, created_at, updated_at) VALUES "
                         "(?, 12, 1, 10, 10, 20, 20, 0.5, 'owlv2', 'needs_review', 5, 't', 't')",
                         (aid,))
    conn.close()


def _roboflow_ready(app, monkeypatch):
    monkeypatch.setattr(export_api, "roboflow_settings", lambda: FAKE_SETTINGS)


FIXTURE_INBOX = "D:\\autoannotate\\inbox"


def _inbox(app, monkeypatch):
    """AUTOLABEL_IMPORT_DIRS = a temp inbox with the fixture's .pt and metrics json; the
    fixture's D:\\autoannotate\\inbox paths are swapped for it."""
    inbox = (Path(app.state.data_dir).parent / "inbox").resolve()
    inbox.mkdir(parents=True, exist_ok=True)
    (inbox / "eot_crane_segment_best.pt").write_bytes(fake_pt())
    (inbox / "eot_crane_segment_metrics.json").write_text(json.dumps(metrics_file()),
                                                          encoding="utf-8")
    monkeypatch.setenv("AUTOLABEL_IMPORT_DIRS", str(inbox))
    return {FIXTURE_INBOX: str(inbox)}


# (tool, example index) -> the example's "state"
STATES = {
    ("run_detection", 0): OWL,
    ("run_detection", 1): {**OWL, "setup": lambda app, mp: add_images(app, 309)},
    ("rerun_examples", 0): {"detectors": ("yoloe",), "modes": {"yoloe": ["examples"]},
                            "setup": _delete_job_12},
    ("set_annotation_status", 1): {"setup": _boxes_201_212},
    ("push_to_roboflow", 0): {"setup": _roboflow_ready},
    ("push_to_roboflow", 1): {"setup": _roboflow_ready},
    ("start_training", 1): {"app": {"cuda_available": lambda: False}},
    ("import_weights", 0): {"setup": _inbox, "app": {"inspect_weights": fake_inspect}},
    ("import_weights", 1): {"setup": _inbox, "app": {"inspect_weights": fake_inspect}},
}
EXAMPLES = [(tool, i, ex) for tool, items in TOOLS_FIXTURE["examples"].items()
            for i, ex in enumerate(items)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path)


@pytest.fixture
def started(tmp_path, env):
    """An app with the fixture data; yields (app, client)."""
    app = make_app(tmp_path)
    with open_client(app) as c:
        load_fixture_data(app)
        yield app, c


def call(app, name, args, user="tester", channel="mcp", button=False):
    return call_tool(ToolContext(app.state, channel, user, button=button), name, args)


# ------------------------------------------------------------------ fixture examples

def test_every_tool_has_an_example():
    names = [t["name"] for t in TOOLS_FIXTURE["tools"]]
    assert list(TOOLS_FIXTURE["examples"]) == names
    assert len(READ_TOOLS) == 8 and len(WRITE_TOOLS) == 11


@pytest.mark.parametrize("tool,index,example", EXAMPLES,
                         ids=[f"{t}-{i}" for t, i, _ in EXAMPLES])
def test_fixture_example(tmp_path, monkeypatch, env, tool, index, example):
    state = STATES.get((tool, index), {})
    if "state" in example:  # (the push examples also need Roboflow .env settings, faked here)
        assert state, "every example with a 'state' needs a STATES entry"
    app = make_app(tmp_path, detectors=state.get("detectors", ("fake",)), modes=state.get("modes"),
                   **state.get("app", {}))
    want, args = example["output"], example["input"]
    with open_client(app):
        load_fixture_data(app)
        if state.get("setup"):
            for old, new in (state["setup"](app, monkeypatch) or {}).items():
                args, want = swap(args, old, new), swap(want, old, new)
        got = call(app, tool, args, user=example.get("user", "tester"))
    assert got["ok"] == want["ok"], got
    if "error" in want:
        assert set(got) == {"ok", "error"} and got["error"]["code"] == want["error"]["code"], got
        assert set(got["error"]) == {"code", "message"}
    elif "needs_confirmation" in want:
        assert set(got) == {"ok", "needs_confirmation"}, got
        nc, wnc = got["needs_confirmation"], want["needs_confirmation"]
        assert set(nc) == set(wnc)
        assert nc["tool"] == wnc["tool"]
        assert nc["details"] == wnc["details"]
        assert nc["arguments"] == wnc["arguments"]
        assert isinstance(nc["effect"], str) and nc["effect"]
    else:
        assert set(got) == {"ok", "data"}
        if tool in READ_TOOLS:  # same data as api_samples.json: the whole value must match
            assert got["data"] == want["data"]
        else:
            same_keys(got["data"], want["data"])


def test_approve_example_values(started):
    app, _ = started
    got = call(app, "set_annotation_status", {"annotation_ids": [101, 102], "action": "approve"},
               user="dhyan")
    assert [(a["id"], a["status"], a["reviewer"]) for a in got["data"]["annotations"]] == [
        (101, "accepted", "claude:dhyan"), (102, "accepted", "claude:dhyan")]


def test_update_example_values(started):
    app, _ = started
    got = call(app, "update_annotation", {"annotation_id": 108, "class_name": "biax"}, user="dhyan")
    a = got["data"]
    assert (a["class_id"], a["class_name"], a["status"], a["reviewer"]) == (
        4, "biax", "edited", "claude:dhyan")


# ------------------------------------------------------------------ input check and order

@pytest.mark.parametrize("name,args", [
    ("project_summary", {}),                                   # missing required
    ("project_summary", {"project_id": "1"}),                  # wrong type
    ("project_summary", {"project_id": 1, "x": 1}),            # additionalProperties
    ("list_images", {"project_id": 1, "limit": 500}),
    ("set_annotation_status", {"annotation_ids": [101, 101], "action": "approve"}),
    ("delete_annotation", {"annotation_id": 106, "confirm": "yes"}),
    ("project_summary", [1]),
])
def test_input_check_is_bad_request(started, name, args):
    app, _ = started
    got = call(app, name, args)
    assert got["ok"] is False and got["error"]["code"] == "bad_request", got


def test_unknown_tool(started):
    app, _ = started
    assert call(app, "drop_tables", {})["error"]["code"] == "bad_request"


def test_would_fail_returns_the_error_not_a_confirmation(tmp_path, env):
    app = make_app(tmp_path, **OWL)
    with open_client(app):
        load_fixture_data(app)
        add_images(app, 309)
        assert call(app, "delete_annotation", {"annotation_id": 999})["error"]["code"] == "not_found"
        assert call(app, "run_detection", {"project_id": 1, "model": "nope"}
                    )["error"]["code"] == "unknown_model"
        got = call(app, "run_detection", {"project_id": 1, "model": "owlv2", "mode": "text",
                                          "masks": "sam2"})
        assert got["error"]["code"] == "bad_request"  # #9 masks rule before the >50 question
        got = call(app, "set_annotation_status",
                   {"annotation_ids": list(range(101, 112)) + [999], "action": "reject"})
        assert got["error"]["code"] == "not_found"


# ------------------------------------------------------------------ confirmations

def test_run_detection_confirm_true_creates_the_job(tmp_path, env):
    app = make_app(tmp_path, **OWL)
    with open_client(app):
        load_fixture_data(app)
        add_images(app, 309)
        got = call(app, "run_detection", {"project_id": 1, "model": "owlv2", "mode": "examples",
                                          "masks": "sam2", "confirm": True})
        assert got["ok"] and got["data"]["total"] == 312
        assert got["data"]["config"] == {"model": "owlv2", "mode": "examples", "image_ids": None,
                                         "options": {}, "masks": "sam2"}


def test_run_detection_50_images_needs_no_confirm(tmp_path, env):
    app = make_app(tmp_path)
    with open_client(app):
        load_fixture_data(app)
        add_images(app, 47)  # 50 in all
        got = call(app, "run_detection", {"project_id": 1, "model": "fake"})
        assert got["ok"] and got["data"]["total"] == 50
        add_images(app, 1, start_id=5000)
        got = call(app, "run_detection", {"project_id": 1, "model": "fake"})
        assert got["needs_confirmation"]["details"] == {"images": 51, "model": "fake",
                                                        "mode": "text", "masks": "none"}


def test_rerun_over_50_todo_images_needs_confirm(tmp_path, env):
    app = make_app(tmp_path, **OWL)
    with open_client(app):
        load_fixture_data(app)
        add_images(app, 60)
        got = call(app, "rerun_examples", {"project_id": 1})
        assert got["needs_confirmation"]["details"] == {"images": 62, "model": "owlv2",
                                                        "mode": "examples", "masks": "sam2"}
        assert rows(app, "SELECT COUNT(*) AS n FROM jobs")[0]["n"] == 6  # nothing created
        got = call(app, "rerun_examples", {"project_id": 1, "confirm": True})
        assert got["ok"] and got["data"]["config"]["rerun_of"] == 12
        assert got["data"]["config"]["masks"] == "sam2"


def test_set_status_10_boxes_runs_11_asks(started):
    app, _ = started
    _boxes_201_212(app, None)
    got = call(app, "set_annotation_status", {"annotation_ids": list(range(201, 211)),
                                              "action": "reject"})
    assert got["ok"] and {a["status"] for a in got["data"]["annotations"]} == {"rejected"}
    got = call(app, "set_annotation_status", {"annotation_ids": [101, *range(201, 211)],
                                              "action": "reject"})
    assert got["needs_confirmation"]["details"] == {"boxes": 11, "action": "reject",
                                                    "by_class": {"person": 11}}
    assert rows(app, "SELECT status FROM annotations WHERE id=101")[0]["status"] == "auto_accepted"
    got = call(app, "set_annotation_status", {"annotation_ids": [101, *range(201, 211)],
                                              "action": "reject", "confirm": True}, user="dhyan")
    assert got["ok"] and len(got["data"]["annotations"]) == 11


def test_set_status_is_all_or_nothing(started):
    app, _ = started
    before = rows(app, "SELECT id, status, reviewer, updated_at FROM annotations ORDER BY id")
    got = call(app, "set_annotation_status", {"annotation_ids": [101, 102, 999],
                                              "action": "approve"})
    assert got["error"]["code"] == "not_found"
    assert rows(app, "SELECT id, status, reviewer, updated_at FROM annotations ORDER BY id") == before


def test_set_status_keeps_input_order(started):
    app, _ = started
    got = call(app, "set_annotation_status", {"annotation_ids": [105, 101, 103],
                                              "action": "approve"})
    assert [a["id"] for a in got["data"]["annotations"]] == [105, 101, 103]


def test_delete_needs_confirm_then_deletes(started):
    app, _ = started
    assert "needs_confirmation" in call(app, "delete_annotation", {"annotation_id": 106})
    assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id=106")[0]["n"] == 1
    got = call(app, "delete_annotation", {"annotation_id": 106, "confirm": False})
    assert "needs_confirmation" in got
    assert call(app, "delete_annotation", {"annotation_id": 106, "confirm": True}) == {
        "ok": True, "data": {"deleted": 106}}
    assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id=106")[0]["n"] == 0


def test_push_checks_come_before_the_confirmation(started, monkeypatch):
    app, _ = started
    monkeypatch.setattr(export_api, "roboflow_settings", lambda: ["ROBOFLOW_WORKSPACE"])
    got = call(app, "push_to_roboflow", {"project_id": 1})
    assert got["error"]["code"] == "conflict" and "ROBOFLOW_WORKSPACE" in got["error"]["message"]
    monkeypatch.setattr(export_api, "roboflow_settings", lambda: FAKE_SETTINGS)
    assert call(app, "push_to_roboflow", {"project_id": 1, "statuses": ["rejected"]}
                )["error"]["code"] == "bad_request"
    got = call(app, "push_to_roboflow", {"project_id": 1, "statuses": ["accepted"]})
    assert got["needs_confirmation"]["details"] == {"images": 1, "statuses": ["accepted"]}
    got = call(app, "push_to_roboflow", {"project_id": 1, "statuses": ["accepted"],
                                         "confirm": True})
    assert got["ok"] and got["data"]["kind"] == "roboflow_push"
    assert got["data"]["config"] == {"statuses": ["accepted"]}


# ------------------------------------------------------------------ other tool rules

def test_add_class_description_defaults_to_the_name(started):
    app, _ = started
    got = call(app, "add_class", {"project_id": 1, "name": "hook"})
    assert got["data"]["class"]["description"] == "hook"
    assert [c["name"] for c in got["data"]["classes"]] == ["person", "rb", "sb", "biax", "hook"]
    # every other class kept as it was
    assert got["data"]["classes"][:4] == call(app, "list_classes", {"project_id": 1}
                                              )["data"]["classes"][:4]


def test_add_class_unknown_project(started):
    app, _ = started
    assert call(app, "add_class", {"project_id": 99, "name": "x"})["error"]["code"] == "not_found"


def test_update_annotation_unknown_class_and_box(started):
    app, _ = started
    assert call(app, "update_annotation", {"annotation_id": 108, "class_name": "crane"}
                )["error"]["code"] == "bad_class_id"
    assert call(app, "update_annotation", {"annotation_id": 999, "status": "accepted"}
                )["error"]["code"] == "not_found"


def test_update_annotation_bbox_follows_13(started):
    app, _ = started
    got = call(app, "update_annotation", {"annotation_id": 101, "bbox": [400, 500, 50, 50]},
               user="dhyan")
    assert got["data"]["bbox"] == [400.0, 500.0, 50.0, 50.0]
    assert got["data"]["status"] == "edited" and got["data"]["reviewer"] == "claude:dhyan"
    got = call(app, "update_annotation", {"annotation_id": 101, "bbox": [1, 1, 0, 5]})
    assert got["error"]["code"] == "bad_request"


def test_set_hook_point_unknown_image(started):
    app, _ = started
    got = call(app, "set_hook_point", {"image_id": 999, "x": 1, "y": 1})
    assert got["error"]["code"] == "bad_image_ids"


def test_get_job_and_annotations_not_found(started):
    app, _ = started
    assert call(app, "get_job", {"job_id": 999})["error"]["code"] == "not_found"
    assert call(app, "get_annotations", {"image_id": 999})["error"]["code"] == "not_found"
    assert call(app, "list_classes", {"project_id": 99})["error"]["code"] == "not_found"


def test_project_summary_lists_every_class_and_null_latest_job(started):
    app, c = started
    pid = c.post("/api/projects", json={"name": "empty"}).json()["id"]
    c.put(f"/api/projects/{pid}/classes", json={"classes": [{"name": "a", "description": "a"}]})
    got = call(app, "project_summary", {"project_id": pid})["data"]
    assert got["boxes_by_class"] == {"a": {"auto_accepted": 0, "needs_review": 0, "accepted": 0,
                                           "rejected": 0, "edited": 0}}
    assert got["latest_job"] is None
    assert got["images_by_review_status"] == {"todo": 0, "done": 0, "empty": 0}


def test_result_size_limit(started):
    app, _ = started
    add_images(app, 250)
    got = call(app, "list_images", {"project_id": 1, "limit": 200})
    assert got == {"ok": False, "error": {"code": "bad_request",
                                          "message": "result too large; use offset/limit"}}
    assert call(app, "list_images", {"project_id": 1, "limit": 20})["ok"]


def test_too_large_write_result_says_it_was_done(started):
    app, _ = started
    conn = connect(app)
    with conn:
        conn.executemany("INSERT INTO annotations (id, image_id, class_id, x, y, w, h, source, "
                         "status, created_at, updated_at) VALUES (?, 12, 1, 1, 1, 5, 5, 'owlv2', "
                         "'needs_review', 't', 't')", [(i,) for i in range(300, 500)])
    conn.close()
    got = call(app, "set_annotation_status", {"annotation_ids": list(range(300, 500)),
                                              "action": "approve", "confirm": True})
    assert got["error"]["code"] == "bad_request" and "WAS done" in got["error"]["message"]
    assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id >= 300 AND "
                     "status='accepted'")[0]["n"] == 200
    assert rows(app, "SELECT outcome FROM assistant_actions")[-1]["outcome"] == "ok"


# ------------------------------------------------------------------ who did it

def test_write_without_a_name_is_refused_reads_work(started):
    app, _ = started
    for user in (None, ""):
        got = call(app, "delete_annotation", {"annotation_id": 106, "confirm": True}, user=user)
        assert got["error"]["code"] == "bad_request" and "X-Autolabel-User" in got["error"]["message"]
    assert call(app, "project_summary", {"project_id": 1}, user=None)["ok"]
    assert rows(app, "SELECT COUNT(*) AS n FROM annotations WHERE id=106")[0]["n"] == 1
    logged = rows(app, "SELECT actor, outcome, summary FROM assistant_actions")
    assert logged == [{"actor": "claude:?", "outcome": "error",
                       "summary": json.dumps({"error": "bad_request"})}] * 2


def test_reads_are_not_logged_writes_always_are(started):
    app, _ = started
    for name in READ_TOOLS:
        call(app, name, {"project_id": 1} if name not in ("list_projects", "get_annotations",
                                                          "get_job") else
             {"get_annotations": {"image_id": 10}, "get_job": {"job_id": 5}}.get(name, {}))
    assert rows(app, "SELECT * FROM assistant_actions") == []

    call(app, "add_class", {"project_id": 1, "name": "hook"}, user="dhyan")           # ok
    call(app, "delete_annotation", {"annotation_id": 106}, user="dhyan")              # confirm
    call(app, "delete_annotation", {"annotation_id": 999}, user="dhyan")              # error
    call(app, "delete_annotation", {"annotation_id": 106, "confirm": True}, user="dhyan")
    got = rows(app, "SELECT channel, actor, project_id, tool, arguments, outcome, confirmed, "
                    "summary FROM assistant_actions ORDER BY id")
    hook_id = rows(app, "SELECT id FROM classes WHERE name='hook'")[0]["id"]
    assert got == [
        {"channel": "mcp", "actor": "claude:dhyan", "project_id": 1, "tool": "add_class",
         "arguments": json.dumps({"project_id": 1, "name": "hook"}), "outcome": "ok",
         "confirmed": 0, "summary": json.dumps({"class_id": hook_id})},
        {"channel": "mcp", "actor": "claude:dhyan", "project_id": 1, "tool": "delete_annotation",
         "arguments": json.dumps({"annotation_id": 106}), "outcome": "needs_confirmation",
         "confirmed": 0, "summary": None},
        {"channel": "mcp", "actor": "claude:dhyan", "project_id": None,
         "tool": "delete_annotation", "arguments": json.dumps({"annotation_id": 999}),
         "outcome": "error", "confirmed": 0, "summary": json.dumps({"error": "not_found"})},
        {"channel": "mcp", "actor": "claude:dhyan", "project_id": 1, "tool": "delete_annotation",
         "arguments": json.dumps({"annotation_id": 106, "confirm": True}), "outcome": "ok",
         "confirmed": 1, "summary": json.dumps({"deleted": 106})},
    ]
    assert all(r["created_at"] for r in rows(app, "SELECT created_at FROM assistant_actions"))


def test_chat_rows_drop_confirm_and_mark_the_button(started):
    app, _ = started
    call(app, "delete_annotation", {"annotation_id": 106, "confirm": True}, channel="chat",
         user="dhyan", button=True)
    got = rows(app, "SELECT channel, arguments, confirmed, outcome FROM assistant_actions")
    assert got == [{"channel": "chat", "arguments": json.dumps({"annotation_id": 106}),
                    "confirmed": 1, "outcome": "ok"}]


def test_unknown_project_id_logs_null_project(started):
    app, _ = started
    assert call(app, "add_class", {"project_id": 99, "name": "x"})["error"]["code"] == "not_found"
    assert rows(app, "SELECT project_id FROM assistant_actions") == [{"project_id": None}]


def test_reviewer_is_claude_name_for_both_writers(started):
    app, _ = started
    call(app, "set_annotation_status", {"annotation_ids": [102], "action": "reject"}, user="ana b")
    call(app, "update_annotation", {"annotation_id": 108, "status": "accepted"}, user="dhyan")
    assert rows(app, "SELECT id, reviewer FROM annotations WHERE id IN (102, 108) ORDER BY id") == [
        {"id": 102, "reviewer": "claude:ana b"}, {"id": 108, "reviewer": "claude:dhyan"}]


# ------------------------------------------------------------------ #35 downloads

def test_train_package_download(started):
    app, c = started
    got = call(app, "make_train_package", {"project_id": 1, "task": "segment"})
    data = got["data"]
    assert data["file_name"] == "eot_crane_train_segment.zip"
    assert data["download_url"].startswith("/api/assistant/downloads/")
    r = c.get(data["download_url"])
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert 'filename="eot_crane_train_segment.zip"' in r.headers["content-disposition"]
    assert len(r.content) == data["size_bytes"]
    with zipfile.ZipFile(BytesIO(r.content)) as zf:
        assert json.loads(zf.read("split.json")) == data["split"]
    expires = datetime.strptime(data["expires_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    assert timedelta(hours=23, minutes=59) < expires - datetime.now(timezone.utc) <= timedelta(hours=24)


def test_train_package_errors(started):
    app, _ = started
    conn = connect(app)
    with conn:
        conn.execute("UPDATE images SET review_status='todo'")
    conn.close()
    assert call(app, "make_train_package", {"project_id": 1})["error"]["code"] == "conflict"
    assert call(app, "make_train_package", {"project_id": 99})["error"]["code"] == "not_found"


@pytest.mark.parametrize("token", ["nope", "..%2F..%2Fapp.db", "A" * 22])
def test_unknown_download_is_404(started, token):
    _, c = started
    r = c.get(f"/api/assistant/downloads/{token}")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


def test_expired_download_is_404_and_deleted(started):
    app, c = started
    data = call(app, "make_train_package", {"project_id": 1, "task": "detect"})["data"]
    token = data["download_url"].rsplit("/", 1)[1]
    meta = downloads.folder(app.state.data_dir) / f"{token}.json"
    m = json.loads(meta.read_text(encoding="utf-8"))
    m["expires_at"] = "2020-01-01T00:00:00Z"
    meta.write_text(json.dumps(m), encoding="utf-8")
    assert c.get(data["download_url"]).status_code == 404
    assert not (downloads.folder(app.state.data_dir) / f"{token}.zip").exists()


def test_old_downloads_deleted_on_new_package_and_at_startup(tmp_path, env):
    app = make_app(tmp_path)
    with open_client(app):
        load_fixture_data(app)
        first = call(app, "make_train_package", {"project_id": 1})["data"]
        token = first["download_url"].rsplit("/", 1)[1]
        meta = downloads.folder(app.state.data_dir) / f"{token}.json"
        m = json.loads(meta.read_text(encoding="utf-8"))
        m["expires_at"] = "2020-01-01T00:00:00Z"
        meta.write_text(json.dumps(m), encoding="utf-8")
        second = call(app, "make_train_package", {"project_id": 1})["data"]
        assert not meta.exists()
        token2 = second["download_url"].rsplit("/", 1)[1]
        meta2 = downloads.folder(app.state.data_dir) / f"{token2}.json"
        m2 = json.loads(meta2.read_text(encoding="utf-8"))
        m2["expires_at"] = "2020-01-01T00:00:00Z"
        meta2.write_text(json.dumps(m2), encoding="utf-8")
    app2 = make_app(tmp_path)  # same data folder, new start
    with open_client(app2):
        assert list(downloads.folder(app2.state.data_dir).glob("*")) == []


# ------------------------------------------------------------------ start_training, import_weights

def test_start_training_local_errors_without_a_confirmation(tmp_path, env):
    for cuda, code in ((False, "no_gpu"), (True, "not_implemented")):
        app = make_app(tmp_path / str(cuda), cuda_available=lambda cuda=cuda: cuda)
        with open_client(app):
            load_fixture_data(app)
            got = call(app, "start_training", {"project_id": 1, "target": "local",
                                               "confirm": True})
            assert got["error"]["code"] == code
            got = call(app, "start_training", {"project_id": 1, "target": "local"})
            assert got["error"]["code"] == code and "needs_confirmation" not in got
            assert [r["outcome"] for r in rows(app, "SELECT outcome FROM assistant_actions")] == [
                "error", "error"]


def test_start_training_checks(tmp_path, env, monkeypatch):
    app = make_app(tmp_path)
    with open_client(app):
        load_fixture_data(app)
        assert call(app, "start_training", {"project_id": 99, "target": "colab"}
                    )["error"]["code"] == "not_found"
        got = call(app, "start_training", {"project_id": 1, "target": "colab", "task": "detect",
                                           "base_weights": "yolo11s-seg.pt"})
        assert got["error"]["code"] == "bad_request"
        assert call(app, "start_training", {"project_id": 1, "target": "colab", "imgsz": 330}
                    )["error"]["code"] == "bad_request"
        got = call(app, "start_training", {"project_id": 1, "target": "colab"})
        assert got["ok"] and got["data"]["package"]["download_url"].startswith(
            "/api/assistant/downloads/")
        row = rows(app, "SELECT tool, outcome, summary FROM assistant_actions")[-1]
        assert (row["tool"], row["outcome"]) == ("start_training", "ok")
        assert json.loads(row["summary"])["target"] == "colab"


@pytest.fixture
def importing(tmp_path, env, monkeypatch):
    """(app, inbox) with AUTOLABEL_IMPORT_DIRS = inbox and a spying fake inspect_weights."""
    inbox = (tmp_path / "inbox").resolve()
    inbox.mkdir()
    monkeypatch.setenv("AUTOLABEL_IMPORT_DIRS", str(inbox))
    seen = []

    def inspect(path, model=None):
        seen.append(path)
        return fake_inspect(path, model)

    app = make_app(tmp_path, inspect_weights=inspect)
    with open_client(app):
        load_fixture_data(app)
        yield app, inbox, seen


def test_import_weights_path_errors_come_before_the_confirmation(importing, tmp_path):
    app, inbox, seen = importing
    outside = tmp_path / "evil.pt"
    outside.write_bytes(fake_pt())
    got = call(app, "import_weights", {"project_id": 1, "path": str(outside), "task": "segment"})
    assert got["error"]["code"] == "path_not_allowed"
    got = call(app, "import_weights", {"project_id": 1, "path": str(inbox / "none.pt"),
                                       "task": "segment"})
    assert got["error"]["code"] == "not_found"
    assert call(app, "import_weights", {"project_id": 99, "path": str(inbox / "none.pt"),
                                        "task": "segment"})["error"]["code"] == "not_found"
    assert seen == []


def test_import_weights_always_confirms_and_loads_only_after(importing):
    app, inbox, seen = importing
    det = inbox / "det.pt"
    det.write_bytes(fake_pt(task="detect"))
    got = call(app, "import_weights", {"project_id": 1, "path": str(det), "task": "segment"})
    nc = got["needs_confirmation"]
    assert nc["details"] == {"path": str(det), "task": "segment", "metrics_path": None,
                             "project_id": 1}
    assert "runs code" in nc["effect"] and seen == []
    got = call(app, "import_weights", {"project_id": 1, "path": str(det), "task": "segment",
                                       "confirm": True})
    assert got["error"]["code"] == "task_mismatch" and len(seen) == 1
    good = inbox / "good.pt"
    good.write_bytes(fake_pt())
    got = call(app, "import_weights", {"project_id": 1, "path": str(good), "task": "segment",
                                       "confirm": True}, user="dhyan")
    assert got["ok"] and got["data"]["name"] == "eot_crane_v2" and good.exists()
    assert [m["name"] for m in call(app, "list_models", {"project_id": 1})["data"]["trained"]] == [
        "eot_crane_v1", "eot_crane_v2"]
    last = rows(app, "SELECT actor, outcome, confirmed FROM assistant_actions")[-1]
    assert last == {"actor": "claude:dhyan", "outcome": "ok", "confirmed": 1}


def test_import_weights_bad_metrics_before_the_confirmation(importing):
    app, inbox, seen = importing
    (inbox / "x_best.pt").write_bytes(fake_pt())
    (inbox / "x_metrics.json").write_text(json.dumps(metrics_file("detect")), encoding="utf-8")
    got = call(app, "import_weights", {"project_id": 1, "path": str(inbox / "x_best.pt"),
                                       "task": "segment"})
    assert got["error"]["code"] == "bad_metrics" and seen == []


def test_new_tool_annotations_are_not_destructive():
    from app.assistant.mcp_server import tool_annotations
    for name in ("start_training", "import_weights"):
        a = tool_annotations(name, "write")
        assert a.read_only_hint is False and a.destructive_hint is False
