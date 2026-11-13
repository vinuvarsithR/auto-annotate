"""#27 - #30 hook points + settings, their migration, #18 cleanup and the #9 `masks` field (API side).
Worker side (PostProcess, result.hook) is in tests/test_worker_hook.py."""

import json
import sqlite3

import pytest

from app.db import db
from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_api_helpers import CRANE_CLASSES, image_bytes, make_client

RB = "rb_load__cam0_2026-09-29_18-34-11"
BIAX = "biax__cam0_2026-09-29_18-48-25"
SB = "sb_evener__cam0_2026-09-29_18-57-19"
NAMES = [f"{RB}_00049.jpg", f"{RB}_00050.jpg", f"{BIAX}_00290.jpg", f"{SB}_00010.jpg"]
OWL_MODES = {"owlv2": ["text", "examples"], "fake": ["text", "examples"]}


@pytest.fixture
def client(tmp_path):
    c, _ = make_client(tmp_path, detectors=("pipeline", "fake", "owlv2"), modes=OWL_MODES)
    with c:
        yield c


def _project(c, names=NAMES, classes=True):
    pid = c.post("/api/projects", json={"name": "crane"}).json()["id"]
    files = [("files", (n, image_bytes(), "image/jpeg")) for n in names]
    imgs = c.post(f"/api/projects/{pid}/images", files=files).json()["images"] if names else []
    if classes:
        c.put(f"/api/projects/{pid}/classes", json={"classes": CRANE_CLASSES})
    return pid, {i["file_name"]: i["id"] for i in imgs}


def _err(r, status, code):
    assert r.status_code == status, r.text
    assert r.json()["error"]["code"] == code
    return r.json()["error"]["message"]


# ------------------------------------------------------------------ #27 GET

def test_get_defaults_and_sequences(client):
    pid, _ = _project(client)
    body = client.get(f"/api/projects/{pid}/hook").json()
    assert body == {"load_classes": ["rb", "sb", "biax"], "load_classes_default": True,
                    "radius": 130.0, "radius_default": True, "points": [],
                    "sequences": [{"sequence": BIAX, "images": 1, "has_point": False},
                                  {"sequence": RB, "images": 2, "has_point": False},
                                  {"sequence": SB, "images": 1, "has_point": False}]}
    same_keys(body, FIXTURES["GET /api/projects/1/hook"])


def test_get_empty_project(client):
    pid, _ = _project(client, names=[])
    assert client.get(f"/api/projects/{pid}/hook").json()["sequences"] == []


def test_get_unknown_project_is_404(client):
    _err(client.get("/api/projects/99/hook"), 404, "not_found")


# ------------------------------------------------------------------ #28 PUT

def test_put_sets_point_for_the_images_video(client):
    pid, ids = _project(client)
    r = client.put(f"/api/projects/{pid}/hook/points",
                   json={"image_id": ids[NAMES[0]], "x": 616, "y": 347})
    assert r.status_code == 200, r.text
    point = r.json()
    same_keys(point, FIXTURES["PUT /api/projects/1/hook/points"])
    assert (point["sequence"], point["x"], point["y"], point["image_id"]) == (
        RB, 616.0, 347.0, ids[NAMES[0]])
    assert isinstance(point["x"], float)
    body = client.get(f"/api/projects/{pid}/hook").json()
    assert body["points"] == [point]
    assert [s["has_point"] for s in body["sequences"]] == [False, True, False]


def test_put_on_another_frame_replaces_the_point(client):
    pid, ids = _project(client)
    client.put(f"/api/projects/{pid}/hook/points", json={"image_id": ids[NAMES[0]], "x": 1, "y": 2})
    client.put(f"/api/projects/{pid}/hook/points",
               json={"image_id": ids[NAMES[1]], "x": 3.5, "y": 4})
    points = client.get(f"/api/projects/{pid}/hook").json()["points"]
    assert [(p["sequence"], p["x"], p["y"], p["image_id"]) for p in points] == [
        (RB, 3.5, 4.0, ids[NAMES[1]])]


def test_put_points_sorted_by_sequence(client):
    pid, ids = _project(client)
    for name in (NAMES[3], NAMES[0], NAMES[2]):
        client.put(f"/api/projects/{pid}/hook/points", json={"image_id": ids[name], "x": 5, "y": 5})
    points = client.get(f"/api/projects/{pid}/hook").json()["points"]
    assert [p["sequence"] for p in points] == [BIAX, RB, SB]


@pytest.mark.parametrize("x,y", [(0, 0), (1280, 720)])
def test_put_edges_are_inside(client, x, y):
    pid, ids = _project(client)
    r = client.put(f"/api/projects/{pid}/hook/points", json={"image_id": ids[NAMES[0]], "x": x, "y": y})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("x,y", [(-0.1, 10), (1280.5, 10), (10, -1), (10, 721)])
def test_put_outside_the_image_is_400(client, x, y):
    pid, ids = _project(client)
    msg = _err(client.put(f"/api/projects/{pid}/hook/points",
                          json={"image_id": ids[NAMES[0]], "x": x, "y": y}), 400, "bad_request")
    assert msg == "x must be in 0..1280 and y in 0..720"


@pytest.mark.parametrize("body", [
    {"x": 1, "y": 2},                                  # missing image_id
    {"image_id": 1, "y": 2},                           # missing x
    {"image_id": 1, "x": 1, "y": 2, "z": 3},           # extra field
    {"image_id": 1, "x": "1", "y": 2},                 # string
    {"image_id": 1, "x": True, "y": 2},                # bool
    {"image_id": 1, "x": None, "y": 2},
    {"image_id": "1", "x": 1, "y": 2},
    [1, 2],
], ids=["no_image", "no_x", "extra", "str_x", "bool_x", "null_x", "str_id", "list"])
def test_put_bad_body_is_400(client, body):
    pid, ids = _project(client)
    if isinstance(body, dict) and body.get("image_id") == 1:
        body = {**body, "image_id": ids[NAMES[0]]}
    _err(client.put(f"/api/projects/{pid}/hook/points", json=body), 400, "bad_request")


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity"])
def test_put_non_finite_is_400(client, value):
    pid, ids = _project(client)
    raw = '{"image_id": %d, "x": %s, "y": 2}' % (ids[NAMES[0]], value)
    _err(client.put(f"/api/projects/{pid}/hook/points", content=raw,
                    headers={"content-type": "application/json"}), 400, "bad_request")


def test_put_image_of_another_project_is_bad_image_ids(client):
    pid, _ = _project(client)
    other = client.post("/api/projects", json={"name": "other"}).json()["id"]
    oid = client.post(f"/api/projects/{other}/images",
                      files=[("files", ("a.jpg", image_bytes(), "image/jpeg"))]
                      ).json()["images"][0]["id"]
    _err(client.put(f"/api/projects/{pid}/hook/points", json={"image_id": oid, "x": 1, "y": 1}),
         400, "bad_image_ids")
    _err(client.put(f"/api/projects/{pid}/hook/points", json={"image_id": 9999, "x": 1, "y": 1}),
         400, "bad_image_ids")


def test_put_unknown_project_is_404_before_bad_body(client):
    _err(client.put("/api/projects/99/hook/points", json={"nonsense": 1}), 404, "not_found")


def test_point_survives_image_delete_with_null_image_id(client):
    pid, ids = _project(client)
    client.put(f"/api/projects/{pid}/hook/points", json={"image_id": ids[NAMES[2]], "x": 1, "y": 1})
    conn = db.connect(client.app.state.db_path)
    try:
        with conn:
            conn.execute("DELETE FROM images WHERE id=?", (ids[NAMES[2]],))
    finally:
        conn.close()
    body = client.get(f"/api/projects/{pid}/hook").json()
    assert body["points"][0]["sequence"] == BIAX and body["points"][0]["image_id"] is None
    assert BIAX not in [s["sequence"] for s in body["sequences"]]  # listed in points only


# ------------------------------------------------------------------ #29 DELETE

def test_delete_point(client):
    pid, ids = _project(client)
    client.put(f"/api/projects/{pid}/hook/points", json={"image_id": ids[NAMES[0]], "x": 1, "y": 1})
    r = client.delete(f"/api/projects/{pid}/hook/points/{RB}")
    assert r.status_code == 204 and r.content == b""
    assert client.get(f"/api/projects/{pid}/hook").json()["points"] == []


def test_delete_url_encoded_sequence(client):
    pid, ids = _project(client, names=["crane cam #1_00001.jpg"])
    point = client.put(f"/api/projects/{pid}/hook/points",
                       json={"image_id": ids["crane cam #1_00001.jpg"], "x": 1, "y": 1}).json()
    assert point["sequence"] == "crane cam #1"
    assert client.delete(f"/api/projects/{pid}/hook/points/crane%20cam%20%231").status_code == 204


def test_delete_missing_point_is_404(client):
    pid, _ = _project(client)
    msg = _err(client.delete(f"/api/projects/{pid}/hook/points/{BIAX}"), 404, "not_found")
    assert msg == FIXTURES["error example (404 not_found, #29 no hook point)"]["error"]["message"]


def test_delete_unknown_project_is_404(client):
    _err(client.delete(f"/api/projects/99/hook/points/{RB}"), 404, "not_found")


def test_points_are_per_project(client):
    pid, ids = _project(client)
    client.put(f"/api/projects/{pid}/hook/points", json={"image_id": ids[NAMES[0]], "x": 1, "y": 1})
    other = client.post("/api/projects", json={"name": "other"}).json()["id"]
    assert client.get(f"/api/projects/{other}/hook").json()["points"] == []
    _err(client.delete(f"/api/projects/{other}/hook/points/{RB}"), 404, "not_found")


# ------------------------------------------------------------------ #30 PATCH

def test_patch_both_and_reset(client):
    pid, _ = _project(client)
    r = client.patch(f"/api/projects/{pid}/hook", json={"load_classes": ["sb", "rb"], "radius": 120})
    assert r.status_code == 200, r.text
    body = r.json()
    same_keys(body, FIXTURES["PATCH /api/projects/1/hook"])
    assert (body["load_classes"], body["load_classes_default"], body["radius"],
            body["radius_default"]) == (["sb", "rb"], False, 120.0, False)
    assert isinstance(body["radius"], float)

    body = client.patch(f"/api/projects/{pid}/hook", json={"radius": None}).json()
    assert (body["load_classes"], body["radius"], body["radius_default"]) == (["sb", "rb"], 130.0, True)
    body = client.patch(f"/api/projects/{pid}/hook", json={"load_classes": None}).json()
    assert (body["load_classes"], body["load_classes_default"]) == (["rb", "sb", "biax"], True)


def test_patch_empty_list_turns_filter_off_and_is_not_default(client):
    pid, _ = _project(client)
    body = client.patch(f"/api/projects/{pid}/hook", json={"load_classes": []}).json()
    assert (body["load_classes"], body["load_classes_default"]) == ([], False)


def test_patch_empty_body_changes_nothing(client):
    pid, _ = _project(client)
    client.patch(f"/api/projects/{pid}/hook", json={"radius": 99.5})
    body = client.patch(f"/api/projects/{pid}/hook", json={}).json()
    assert body["radius"] == 99.5 and body["load_classes_default"] is True


@pytest.mark.parametrize("body", [
    {"load_classes": ["rb", "crane"]},     # unknown name
    {"load_classes": ["rb", "rb"]},        # repeated
    {"load_classes": "rb"},
    {"load_classes": [1]},
    {"radius": 0}, {"radius": -5}, {"radius": "10"}, {"radius": True},
    {"radius": 10, "other": 1},            # extra field
    [1],
])
def test_patch_bad_body_is_400_and_changes_nothing(client, body):
    pid, _ = _project(client)
    _err(client.patch(f"/api/projects/{pid}/hook", json=body), 400, "bad_request")
    body = client.get(f"/api/projects/{pid}/hook").json()
    assert body["load_classes_default"] and body["radius_default"]


def test_patch_infinite_radius_is_400(client):
    pid, _ = _project(client)
    _err(client.patch(f"/api/projects/{pid}/hook", content='{"radius": Infinity}',
                      headers={"content-type": "application/json"}), 400, "bad_request")


def test_patch_unknown_project_is_404(client):
    _err(client.patch("/api/projects/99/hook", json={"radius": 10}), 404, "not_found")


def test_class_delete_keeps_the_setting(client):
    pid, _ = _project(client)
    client.patch(f"/api/projects/{pid}/hook", json={"load_classes": ["rb", "sb"]})
    keep = [c for c in CRANE_CLASSES if c["name"] != "sb"]
    assert client.put(f"/api/projects/{pid}/classes", json={"classes": keep}).status_code == 200
    assert client.get(f"/api/projects/{pid}/hook").json()["load_classes"] == ["rb", "sb"]


def test_project_json_has_no_hook_columns(client):
    pid, _ = _project(client)
    client.patch(f"/api/projects/{pid}/hook", json={"radius": 10})
    same_keys(client.get(f"/api/projects/{pid}").json(), FIXTURES["GET /api/projects/1"])


# ------------------------------------------------------------------ #18 and migration

def test_project_delete_removes_hook_points(client):
    pid, ids = _project(client)
    client.put(f"/api/projects/{pid}/hook/points", json={"image_id": ids[NAMES[0]], "x": 1, "y": 1})
    assert client.delete(f"/api/projects/{pid}").status_code == 204
    conn = db.connect(client.app.state.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM hook_points").fetchone()[0] == 0
    finally:
        conn.close()


def _old_db(path):
    """A database from before #27: no hook_points table, projects without the two columns."""
    conn = sqlite3.connect(path)
    try:
        conn.executescript(db.SCHEMA.read_text(encoding="utf-8").split(
            "CREATE TABLE hook_points")[0])
        conn.execute("CREATE TABLE p2 (id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE, "
                     "created_at TEXT NOT NULL)")
        conn.execute("INSERT INTO p2 VALUES (1, 'old', '2026-10-30T10:00:00Z')")
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP TABLE projects")
        conn.execute("ALTER TABLE p2 RENAME TO projects")
        conn.commit()
    finally:
        conn.close()


def test_migrate_adds_table_and_columns_to_an_old_db(tmp_path):
    path = tmp_path / "app.db"
    _old_db(path)
    conn = sqlite3.connect(path)
    assert {r[1] for r in conn.execute("PRAGMA table_info(projects)")} == {"id", "name", "created_at"}
    assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='hook_points'").fetchone()
    conn.close()

    db.init_db(path)
    db.init_db(path)  # second run: nothing to do, no error
    conn = db.connect(path)
    try:
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(projects)")]
        assert cols == ["id", "name", "created_at", "load_classes", "hook_radius"]
        assert dict(conn.execute("SELECT * FROM projects").fetchone()) == {
            "id": 1, "name": "old", "created_at": "2026-10-30T10:00:00Z",
            "load_classes": None, "hook_radius": None}
        hp = [r["name"] for r in conn.execute("PRAGMA table_info(hook_points)")]
        assert hp == ["project_id", "sequence", "x", "y", "image_id", "updated_at"]
        assert db.hook_settings(conn, 1) == {"load_classes": ["rb", "sb", "biax"],
                                             "load_classes_default": True, "radius": 130.0,
                                             "radius_default": True}
    finally:
        conn.close()


def test_fresh_db_has_the_hook_table(tmp_path):
    db.init_db(tmp_path / "app.db")
    conn = db.connect(tmp_path / "app.db")
    try:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='hook_points'").fetchone()
    finally:
        conn.close()


# ------------------------------------------------------------------ #9 masks (API checks)

def _post(c, pid, **body):
    return c.post(f"/api/projects/{pid}/jobs", json=body)


@pytest.mark.parametrize("masks", ["sam2", "none"])
def test_masks_is_stored_in_config(client, masks):
    pid, _ = _project(client)
    r = _post(client, pid, model="owlv2", mode="examples", masks=masks)
    assert r.status_code == 202, r.text
    assert r.json()["config"] == {"model": "owlv2", "mode": "examples", "image_ids": None,
                                  "options": {}, "masks": masks}


def test_fixture_request_matches(client):
    pid, _ = _project(client)
    body = FIXTURES["POST /api/projects/1/jobs (owlv2 examples + sam2) request"]
    r = _post(client, pid, **body)
    assert r.status_code == 202, r.text
    assert r.json()["config"] == FIXTURES["GET /api/jobs/12 (detect, examples, owlv2 + sam2)"]["config"]


def test_absent_masks_keeps_the_old_config_shape(client):
    pid, _ = _project(client)
    config = _post(client, pid, model="owlv2", mode="examples").json()["config"]
    assert "masks" not in config


@pytest.mark.parametrize("masks", ["SAM2", "", None, 1, True, ["sam2"]])
def test_bad_masks_value_is_400(client, masks):
    pid, _ = _project(client)
    _err(_post(client, pid, model="owlv2", mode="examples", masks=masks), 400, "bad_request")


@pytest.mark.parametrize("body", [
    {"model": "owlv2", "mode": "text", "masks": "sam2"},
    {"model": "owlv2", "mode": "text", "masks": "none"},
    {"model": "fake", "mode": "examples", "masks": "none"},
    {"model": "pipeline", "masks": "sam2"},
])
def test_masks_on_other_model_or_mode_is_400(client, body):
    pid, _ = _project(client)
    _err(_post(client, pid, **body), 400, "bad_request")


def test_owlv2_default_mode_is_text_so_masks_is_400(client):
    pid, _ = _project(client)
    _err(_post(client, pid, model="owlv2", masks="sam2"), 400, "bad_request")


def test_masks_on_a_trained_model_is_400(client):
    pid, _ = _project(client)
    conn = db.connect(client.app.state.db_path)
    try:
        with conn:
            conn.execute("INSERT INTO trained_models (project_id, name, version, task, class_names, "
                         "file_name, path, size_bytes, uploaded_at) VALUES (?, 'crane_v1', 1, "
                         "'detect', '[]', 'best.pt', 'x', 1, 't')", (pid,))
    finally:
        conn.close()
    _err(_post(client, pid, model="crane_v1", masks="none"), 400, "bad_request")


def test_check_order_unknown_mode_before_masks_before_no_classes(client):
    pid, _ = _project(client, classes=False)
    _err(_post(client, pid, model="owlv2", mode="bogus", masks="bad"), 400, "unknown_mode")
    _err(_post(client, pid, model="owlv2", mode="examples", masks="bad"), 400, "bad_request")
    _err(_post(client, pid, model="owlv2", mode="examples", masks="sam2"), 400, "no_classes")


def test_masks_error_matches_fixture_shape(client):
    pid, _ = _project(client)
    r = _post(client, pid, model="fake", mode="examples", masks="sam2")
    same_keys(r.json(), FIXTURES["error example (400 bad_request, #9 masks needs owlv2 examples)"])


def test_bad_masks_creates_no_job(client):
    pid, _ = _project(client)
    _post(client, pid, model="fake", masks="sam2")
    conn = db.connect(client.app.state.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    finally:
        conn.close()


# ------------------------------------------------------------------ rerun copies masks

def _source_job(c, pid, config):
    conn = db.connect(c.app.state.db_path)
    try:
        db.create_job(conn, pid, "detect", config, 1)
    finally:
        conn.close()


@pytest.mark.parametrize("masks", ["sam2", "none"])
def test_rerun_copies_masks(client, masks):
    pid, _ = _project(client)
    _source_job(client, pid, {"model": "owlv2", "mode": "examples", "image_ids": None,
                              "options": {}, "masks": masks})
    r = _post(client, pid, rerun=True)
    assert r.status_code == 202, r.text
    assert r.json()["config"]["masks"] == masks


def test_rerun_of_a_job_without_masks_has_no_masks(client):
    pid, _ = _project(client)
    _source_job(client, pid, {"model": "owlv2", "mode": "examples", "image_ids": None,
                              "options": {}})
    config = _post(client, pid, rerun=True).json()["config"]
    assert "masks" not in config


def test_rerun_with_masks_field_is_400(client):
    pid, _ = _project(client)
    _source_job(client, pid, {"model": "owlv2", "mode": "examples", "image_ids": None,
                              "options": {}})
    msg = _err(_post(client, pid, rerun=True, masks="sam2"), 400, "bad_request")
    assert "masks" in msg
