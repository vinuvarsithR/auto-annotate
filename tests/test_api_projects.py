"""Tests for endpoints 1-3 and the error shape."""

from tests.test_api_helpers import make_client


def test_create_list_get(tmp_path):
    with make_client(tmp_path)[0] as c:
        r = c.post("/api/projects", json={"name": "eot_crane"})
        assert r.status_code == 201
        p = r.json()
        assert p["name"] == "eot_crane"
        assert p["counts"] == {"images": 0, "auto_accepted": 0, "needs_review": 0,
                               "accepted": 0, "rejected": 0, "edited": 0}
        assert c.get("/api/projects").json()["projects"] == [p]
        assert c.get(f"/api/projects/{p['id']}").json() == p


def test_duplicate_name_409(tmp_path):
    with make_client(tmp_path)[0] as c:
        c.post("/api/projects", json={"name": "a"})
        r = c.post("/api/projects", json={"name": "a"})
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "name_taken"


def test_missing_project_404_shape(tmp_path):
    with make_client(tmp_path)[0] as c:
        r = c.get("/api/projects/99")
        assert r.status_code == 404
        assert r.json() == {"error": {"code": "not_found", "message": "project 99 not found"}}


def test_bad_body_400_shape(tmp_path):
    with make_client(tmp_path)[0] as c:
        r = c.post("/api/projects", json={"nope": 1})
        assert r.status_code == 400
        assert r.json()["error"]["code"] == "bad_request"
        r = c.post("/api/projects", json={"name": "   "})
        assert r.status_code == 400


def test_unknown_api_path_404_shape(tmp_path):
    with make_client(tmp_path)[0] as c:
        r = c.get("/api/nothing_here")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "not_found"
