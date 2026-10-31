"""Tests for endpoints 1-3, 17 (rename), 18 (delete) and the error shape."""

from app.db import db
from tests.test_api_helpers import make_client, run_job, setup_project


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


# --- #17 rename -------------------------------------------------------------

def test_rename_project(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=2)
        r = c.patch(f"/api/projects/{s['pid']}", json={"name": "  eot_crane_v2 "})
        assert r.status_code == 200
        p = r.json()
        assert p["name"] == "eot_crane_v2" and p["counts"]["images"] == 2
        assert c.get(f"/api/projects/{s['pid']}").json() == p
        # same name again: no-op 200
        assert c.patch(f"/api/projects/{s['pid']}", json={"name": "eot_crane_v2"}).json() == p


def test_rename_errors(tmp_path):
    with make_client(tmp_path)[0] as c:
        a = c.post("/api/projects", json={"name": "a"}).json()["id"]
        c.post("/api/projects", json={"name": "b"})
        r = c.patch(f"/api/projects/{a}", json={"name": "b"})
        assert (r.status_code, r.json()["error"]["code"]) == (409, "name_taken")
        for body in ({"name": "   "}, {"name": ""}, {}):
            r = c.patch(f"/api/projects/{a}", json=body)
            assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_request")
        r = c.patch("/api/projects/99", json={"name": "x"})
        assert (r.status_code, r.json()["error"]["code"]) == (404, "not_found")
        assert c.get(f"/api/projects/{a}").json()["name"] == "a"


# --- #18 delete -------------------------------------------------------------

def _table_counts(c, pid):
    conn = db.connect(c.app.state.db_path)
    try:
        q = lambda sql: conn.execute(sql, (pid,)).fetchone()[0]
        return {
            "projects": q("SELECT COUNT(*) FROM projects WHERE id=?"),
            "images": q("SELECT COUNT(*) FROM images WHERE project_id=?"),
            "classes": q("SELECT COUNT(*) FROM classes WHERE project_id=?"),
            "jobs": q("SELECT COUNT(*) FROM jobs WHERE project_id=?"),
            "annotations": q("SELECT COUNT(*) FROM annotations a JOIN images i "
                             "ON a.image_id=i.id WHERE i.project_id=?"),
        }
    finally:
        conn.close()


def test_delete_project_removes_rows_and_files(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=2)
        keep = setup_project(c, n_images=1, name="other")
        run_job(c, s["pid"])
        run_job(c, keep["pid"])
        iid = s["image_ids"][0]
        c.post(f"/api/images/{iid}/annotations",
               json={"class_id": s["class_ids"]["biax"], "bbox": [1, 2, 3, 4]})
        folder = tmp_path / "data" / "uploads" / str(s["pid"])
        assert len(list(folder.iterdir())) == 2
        before_keep = _table_counts(c, keep["pid"])

        r = c.delete(f"/api/projects/{s['pid']}")
        assert r.status_code == 204 and r.content == b""
        assert _table_counts(c, s["pid"]) == dict.fromkeys(before_keep, 0)
        assert not folder.exists()
        assert c.get(f"/api/projects/{s['pid']}").status_code == 404
        assert c.get(f"/api/images/{iid}/file").status_code == 404
        # the other project is untouched
        assert _table_counts(c, keep["pid"]) == before_keep
        assert (tmp_path / "data" / "uploads" / str(keep["pid"])).is_dir()
        assert c.get(f"/api/images/{keep['image_ids'][0]}/file").status_code == 200
        r = c.delete(f"/api/projects/{s['pid']}")
        assert (r.status_code, r.json()["error"]["code"]) == (404, "not_found")


def test_delete_project_with_active_job_409(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=2)
        folder = tmp_path / "data" / "uploads" / str(s["pid"])
        # worker is not started in tests, so the job stays queued
        assert c.post(f"/api/projects/{s['pid']}/jobs", json={"model": "pipeline"}).status_code == 202
        before = _table_counts(c, s["pid"])
        r = c.delete(f"/api/projects/{s['pid']}")
        assert r.status_code == 409
        assert r.json() == {"error": {"code": "conflict",
                                      "message": f"project {s['pid']} has a running job"}}
        assert _table_counts(c, s["pid"]) == before
        assert len(list(folder.iterdir())) == 2

        # running counts too
        conn = db.connect(c.app.state.db_path)
        with conn:
            conn.execute("UPDATE jobs SET status='running' WHERE project_id=?", (s["pid"],))
        conn.close()
        assert c.delete(f"/api/projects/{s['pid']}").status_code == 409

        # finished jobs do not block
        conn = db.connect(c.app.state.db_path)
        with conn:
            conn.execute("UPDATE jobs SET status='done' WHERE project_id=?", (s["pid"],))
        conn.close()
        assert c.delete(f"/api/projects/{s['pid']}").status_code == 204


def test_delete_project_without_upload_folder(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = c.post("/api/projects", json={"name": "empty"}).json()["id"]
        assert c.delete(f"/api/projects/{pid}").status_code == 204
