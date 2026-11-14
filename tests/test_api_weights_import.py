"""#38 POST /projects/{pid}/weights/import: only from the import folders (AUTOLABEL_IMPORT_DIRS,
default data/inbox), paths resolved (.. and symlinks cannot leave), sibling _metrics.json picked
up, the original file kept, the #24 checks and storage via store_weights."""

import json
import os
from pathlib import Path

import pytest

from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_assistant_helpers import (
    export_shims, fake_inspect, fake_pt, hermetic_env, load_fixture_data, make_app, metrics_file,
    open_client,
)


class Spy:
    def __init__(self):
        self.inspected = []

    def __call__(self, path, model=None):
        self.inspected.append(path)
        return fake_inspect(path, model)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """(client, inbox, outside folder, spy); AUTOLABEL_IMPORT_DIRS = inbox only."""
    hermetic_env(monkeypatch, tmp_path)
    export_shims(monkeypatch)
    inbox = (tmp_path / "inbox").resolve()
    outside = (tmp_path / "outside").resolve()
    inbox.mkdir()
    outside.mkdir()
    monkeypatch.setenv("AUTOLABEL_IMPORT_DIRS", str(inbox))
    spy = Spy()
    app = make_app(tmp_path, inspect_weights=spy)
    with open_client(app) as c:
        load_fixture_data(app)
        yield c, inbox, outside, spy


def post(c, body, pid=1):
    return c.post(f"/api/projects/{pid}/weights/import", json=body)


def err(r, status, code):
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code, r.text


def put(folder: Path, name: str, data: bytes) -> Path:
    p = folder / name
    p.write_bytes(data)
    return p


def test_import_with_sibling_metrics(env):
    c, inbox, _, _ = env
    src = put(inbox, "eot_crane_segment_best.pt", fake_pt())
    put(inbox, "eot_crane_segment_metrics.json", json.dumps(metrics_file()).encode())
    r = post(c, {"path": str(src), "task": "segment"})
    assert r.status_code == 201, r.text
    m = r.json()
    same_keys(m, FIXTURES["POST /api/projects/1/weights/import"])
    assert (m["name"], m["version"], m["source"]) == ("eot_crane_v2", 2, "colab")
    assert m["file_name"] == "eot_crane_segment_best.pt"
    assert m["metrics"] == metrics_file()["metrics"] and m["split"] == metrics_file()["split"]
    assert src.read_bytes() == fake_pt()  # the original stays
    stored = Path(c.app.state.data_dir) / "weights" / "1" / "eot_crane_v2.pt"
    assert stored.read_bytes() == fake_pt()


def test_import_without_metrics_and_explicit_metrics_path(env):
    c, inbox, _, _ = env
    src = put(inbox, "plain.pt", fake_pt())
    r = post(c, {"path": str(src), "task": "segment", "metrics_path": None})
    assert r.status_code == 201, r.text
    assert (r.json()["metrics"], r.json()["split"]) == (None, None)
    mp = put(inbox, "other.json", json.dumps(metrics_file()).encode())
    r = post(c, {"path": str(src), "task": "segment", "metrics_path": str(mp)})
    assert r.status_code == 201, r.text
    assert r.json()["metrics"] == metrics_file()["metrics"]


def test_outside_the_import_folders_is_refused_before_loading(env):
    c, inbox, outside, spy = env
    src = put(outside, "evil.pt", fake_pt())
    r = post(c, {"path": str(src), "task": "segment"})
    err(r, 400, "path_not_allowed")
    assert "AUTOLABEL_IMPORT_DIRS" in r.json()["error"]["message"]
    dotdot = str(inbox / ".." / "outside" / "evil.pt")
    err(post(c, {"path": dotdot, "task": "segment"}), 400, "path_not_allowed")
    err(post(c, {"path": "evil.pt", "task": "segment"}), 400, "path_not_allowed")  # relative
    err(post(c, {"path": "C:\\Windows\\evil.pt", "task": "segment"}), 400, "path_not_allowed")
    ok = put(inbox, "ok.pt", fake_pt())
    err(post(c, {"path": str(ok), "task": "segment", "metrics_path": str(outside / "m.json")}),
        400, "path_not_allowed")
    assert spy.inspected == []


def test_a_symlink_out_of_the_folder_is_refused(env):
    c, inbox, outside, spy = env
    target = put(outside, "evil.pt", fake_pt())
    link = inbox / "link.pt"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("this OS / user cannot create symlinks")
    err(post(c, {"path": str(link), "task": "segment"}), 400, "path_not_allowed")
    assert spy.inspected == []


def test_a_sibling_metrics_symlink_out_is_refused(env):
    c, inbox, outside, _ = env
    src = put(inbox, "x_best.pt", fake_pt())
    target = put(outside, "m.json", json.dumps(metrics_file()).encode())
    try:
        os.symlink(target, inbox / "x_metrics.json")
    except (OSError, NotImplementedError):
        pytest.skip("this OS / user cannot create symlinks")
    err(post(c, {"path": str(src), "task": "segment"}), 400, "path_not_allowed")


def test_file_rules(env):
    c, inbox, _, spy = env
    err(post(c, {"path": str(inbox / "missing.pt"), "task": "segment"}), 404, "not_found")
    err(post(c, {"path": str(inbox), "task": "segment"}), 404, "not_found")  # a folder
    txt = put(inbox, "w.txt", fake_pt())
    err(post(c, {"path": str(txt), "task": "segment"}), 400, "bad_request")
    empty = put(inbox, "empty.pt", b"")
    err(post(c, {"path": str(empty), "task": "segment"}), 400, "bad_request")
    assert spy.inspected == []


@pytest.mark.parametrize("body", [None, [], {"task": "segment"}, {"path": "", "task": "segment"},
                                  {"path": "x.pt"}, {"path": "x.pt", "task": "classify"},
                                  {"path": "x.pt", "task": "segment", "extra": 1},
                                  {"path": "x.pt", "task": "segment", "metrics_path": 3}])
def test_bad_body(env, body):
    c, *_ = env
    r = c.post("/api/projects/1/weights/import") if body is None else post(c, body)
    err(r, 400, "bad_request")


def test_unknown_project_is_404(env):
    c, inbox, _, _ = env
    src = put(inbox, "a.pt", fake_pt())
    err(post(c, {"path": str(src), "task": "segment"}, pid=99), 404, "not_found")


def test_bad_metrics_and_weights_checks(env):
    c, inbox, _, spy = env
    src = put(inbox, "m_best.pt", fake_pt())
    put(inbox, "m_metrics.json", json.dumps(dict(metrics_file(), format="x")).encode())
    err(post(c, {"path": str(src), "task": "segment"}), 400, "bad_metrics")
    assert spy.inspected == []
    det = put(inbox, "det.pt", fake_pt(task="detect"))
    err(post(c, {"path": str(det), "task": "segment"}), 400, "task_mismatch")
    wrong = put(inbox, "wrong.pt", fake_pt(names=["person"]))
    err(post(c, {"path": str(wrong), "task": "segment"}), 400, "class_mismatch")
    bad = put(inbox, "bad.pt", b"bad bytes")
    err(post(c, {"path": str(bad), "task": "segment"}), 400, "bad_weights")
    assert det.exists() and wrong.exists() and bad.exists()  # originals never deleted
    assert [w["name"] for w in c.get("/api/projects/1/weights").json()["weights"]] == [
        "eot_crane_v1"]


def test_default_import_folder_is_data_inbox(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path)
    export_shims(monkeypatch)
    app = make_app(tmp_path, inspect_weights=fake_inspect)
    with open_client(app) as c:
        load_fixture_data(app)
        inbox = Path(app.state.data_dir) / "inbox"
        assert inbox.is_dir()  # created at startup
        src = put(inbox, "a.pt", fake_pt())
        assert post(c, {"path": str(src.resolve()), "task": "segment"}).status_code == 201
        other = put(tmp_path, "b.pt", fake_pt())
        err(post(c, {"path": str(other.resolve()), "task": "segment"}), 400, "path_not_allowed")


def test_several_import_folders(tmp_path, monkeypatch):
    hermetic_env(monkeypatch, tmp_path)
    export_shims(monkeypatch)
    a, b = (tmp_path / "a").resolve(), (tmp_path / "b b").resolve()
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("AUTOLABEL_IMPORT_DIRS", f"{a} ; {b};relative\\dir")
    app = make_app(tmp_path, inspect_weights=fake_inspect)
    with open_client(app) as c:
        load_fixture_data(app)
        for folder in (a, b):
            src = put(folder, "w.pt", fake_pt())
            assert post(c, {"path": str(src), "task": "segment"}).status_code == 201
