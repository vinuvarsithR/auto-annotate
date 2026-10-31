"""Tests for endpoints 7-8: class list replace-by-name."""

from tests.test_api_helpers import CRANE_CLASSES, make_client, run_job, setup_project


def test_put_then_get(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = c.post("/api/projects", json={"name": "p"}).json()["id"]
        r = c.put(f"/api/projects/{pid}/classes", json={"classes": CRANE_CLASSES})
        assert r.status_code == 200
        cls = r.json()["classes"]
        assert [x["name"] for x in cls] == ["person", "rb", "sb", "biax"]
        assert cls[0]["accept_score"] == 0.5 and cls[1]["accept_score"] is None
        assert c.get(f"/api/projects/{pid}/classes").json() == {"classes": cls}


def test_update_add_delete_by_name_keeps_ids(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = c.post("/api/projects", json={"name": "p"}).json()["id"]
        first = c.put(f"/api/projects/{pid}/classes", json={"classes": CRANE_CLASSES}).json()
        ids = {x["name"]: x["id"] for x in first["classes"]}
        new = [{"name": "person", "description": "a person", "accept_score": 0.7},
               {"name": "hook", "description": "a crane hook"}]
        out = c.put(f"/api/projects/{pid}/classes", json={"classes": new}).json()["classes"]
        by = {x["name"]: x for x in out}
        assert set(by) == {"person", "hook"}
        assert by["person"]["id"] == ids["person"]
        assert by["person"]["description"] == "a person"
        assert by["person"]["accept_score"] == 0.7 and by["person"]["min_score"] is None


def test_removing_class_in_use_is_409_and_changes_nothing(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        run_job(c, s["pid"])  # creates person + rb boxes
        keep = [x for x in CRANE_CLASSES if x["name"] not in ("rb", "sb")]
        r = c.put(f"/api/projects/{s['pid']}/classes", json={"classes": keep})
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "class_in_use"
        assert "rb" in r.json()["error"]["message"]
        names = [x["name"] for x in c.get(f"/api/projects/{s['pid']}/classes").json()["classes"]]
        assert names == ["person", "rb", "sb", "biax"]  # sb not deleted either


def test_duplicate_names_and_bad_scores_400(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = c.post("/api/projects", json={"name": "p"}).json()["id"]
        dup = [{"name": "a", "description": "x"}, {"name": "a", "description": "y"}]
        assert c.put(f"/api/projects/{pid}/classes", json={"classes": dup}).status_code == 400
        bad = [{"name": "a", "description": "x", "min_score": 1.5}]
        assert c.put(f"/api/projects/{pid}/classes", json={"classes": bad}).status_code == 400
