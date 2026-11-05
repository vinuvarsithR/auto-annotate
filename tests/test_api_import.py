"""#22 POST /projects/{pid}/import + the worker's import jobs. Importers are stubbed
(tests/test_api_import_stub.py), so this does not depend on export's code."""

import io
import json
import zipfile

import pytest

from app.db import db
from tests import test_api_import_stub as stub
from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_api_helpers import CRANE_CLASSES, image_bytes, make_client, run_job


def _client(tmp_path):
    c, _ = make_client(tmp_path)
    c.app.state.worker.load_importers = stub.module
    return c


def _zip(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _labels(boxes: dict, fmt="yolo") -> bytes:
    return json.dumps({"format": fmt, "boxes": boxes}).encode()


def _post_zip(c, pid, entries, name="crane_yolo.zip"):
    return c.post(f"/api/projects/{pid}/import",
                  files=[("files", (name, _zip(entries), "application/zip"))])


def _run(c, job):
    assert c.app.state.worker.run_once()
    return c.get(f"/api/jobs/{job['id']}").json()


def _imports_dir(c):
    return c.app.state.data_dir / "imports"


def _crane_project(c):
    pid = _project(c)
    cls = c.put(f"/api/projects/{pid}/classes", json={"classes": CRANE_CLASSES}).json()
    return {"pid": pid, "class_ids": {k["name"]: k["id"] for k in cls["classes"]}}


def _project(c, name="eot_crane"):
    return c.post("/api/projects", json={"name": name}).json()["id"]


# ------------------------------------------------------------------ happy paths

def test_zip_import_adds_images_boxes_and_classes(tmp_path):
    with _client(tmp_path) as c:
        s = _crane_project(c)
        entries = {
            "train/images/f_1.jpg": image_bytes(),
            "train/images/f_2.png": image_bytes("PNG"),
            "README.roboflow.txt": b"hello",
            "labels.json": _labels({
                "train/images/f_1.jpg": [{"class_name": "person", "bbox": [1, 2, 30, 40]},
                                         {"class_name": "hook", "bbox": [5, 5, 10, 10]}],
                "train/images/f_2.png": [{"class_name": "crate", "bbox": [0, 0, 8, 9]},
                                         {"class_name": "hook", "bbox": [1, 1, 2, 2]}]}),
        }
        r = _post_zip(c, s["pid"], entries)
        assert r.status_code == 202, r.text
        job = r.json()
        assert (job["kind"], job["status"], job["total"], job["result"]) == \
            ("import", "queued", 2, None)
        assert job["config"] == {"upload": "zip", "file_name": "crane_yolo.zip"}
        assert (_imports_dir(c) / str(job["id"]) / "upload.zip").is_file()

        done = _run(c, job)
        assert done["status"] == "done", done
        assert (done["done"], done["total"], done["progress"]) == (2, 2, 1.0)
        assert done["result"] == {
            "format": "yolo", "images_added": 2, "duplicates": [], "renamed": [],
            "boxes_by_class": {"person": 1, "hook": 2, "crate": 1},
            "classes_created": ["hook", "crate"],
            "skipped": [{"file_name": "README.roboflow.txt", "reason": "not an image"}]}
        assert not (_imports_dir(c) / str(job["id"])).exists()  # temp folder deleted

        classes = c.get(f"/api/projects/{s['pid']}/classes").json()["classes"]
        assert [k["name"] for k in classes] == ["person", "rb", "sb", "biax", "hook", "crate"]
        assert classes[-1] == {**classes[-1], "description": "crate", "min_score": None,
                               "accept_score": None}

        imgs = c.get(f"/api/projects/{s['pid']}/images").json()["images"]
        assert [i["file_name"] for i in imgs] == ["f_1.jpg", "f_2.png"]
        assert all(i["review_status"] == "todo" for i in imgs)
        anns = c.get(f"/api/images/{imgs[0]['id']}/annotations").json()["annotations"]
        assert [(a["class_name"], a["bbox"]) for a in anns] == \
            [("person", [1, 2, 30, 40]), ("hook", [5, 5, 10, 10])]
        for a in anns:
            assert (a["status"], a["source"], a["score"], a["mask"], a["job_id"], a["reviewer"]) \
                == ("needs_review", "import", None, None, job["id"], None)
        assert c.get(f"/api/images/{imgs[1]['id']}/file").status_code == 200


def test_import_job_stores_box_polygon(tmp_path):
    triangle = [[0.5, 0.25], [0.75, 0.25], [0.625, 0.5]]
    with _client(tmp_path) as c:
        s = _crane_project(c)
        entries = {
            "images/a.jpg": image_bytes(),
            "labels.json": _labels({"images/a.jpg": [
                {"class_name": "rb", "bbox": [640, 180, 320, 180], "polygon": triangle},
                {"class_name": "rb", "bbox": [1, 2, 3, 4], "polygon": None},
                {"class_name": "rb", "bbox": [5, 6, 7, 8]}]}),
        }
        job = _run(c, _post_zip(c, s["pid"], entries).json())
        assert job["status"] == "done" and job["result"]["boxes_by_class"] == {"rb": 3}
        iid = c.get(f"/api/projects/{s['pid']}/images").json()["images"][0]["id"]
        rows = c.get(f"/api/images/{iid}/annotations").json()["annotations"]
        assert [a["polygon"] for a in rows] == [triangle, None, None]
        assert rows[0]["bbox"] == [640.0, 180.0, 320.0, 180.0]
        assert (rows[0]["source"], rows[0]["status"], rows[0]["score"], rows[0]["mask"],
                rows[0]["job_id"]) == ("import", "needs_review", None, None, job["id"])


def test_duplicates_and_renames_follow_rule_4(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        old = image_bytes(colour=(1, 2, 3))
        c.post(f"/api/projects/{pid}/images",
               files=[("files", ("f_1.jpg", old, "image/jpeg")),
                      ("files", ("x.jpg", image_bytes(), "image/jpeg"))])
        same = image_bytes(colour=(9, 9, 9))
        entries = {
            "a/f_1.jpg": old,                 # same bytes as the project's f_1.jpg
            "b/x.jpg": image_bytes(),         # same name, new bytes -> x_2.jpg
            "c/y.jpg": same,
            "d/y.jpg": same,                  # duplicate of an earlier image in this job
            "labels.json": _labels({"a/f_1.jpg": [{"class_name": "person", "bbox": [1, 1, 5, 5]}],
                                    "d/y.jpg": [{"class_name": "dupe", "bbox": [1, 1, 5, 5]}],
                                    "b/x.jpg": [{"class_name": "person", "bbox": [2, 2, 5, 5]}]}),
        }
        res = _run(c, _post_zip(c, pid, entries).json())["result"]
        assert res["images_added"] == 2
        assert res["duplicates"] == ["a/f_1.jpg", "d/y.jpg"]
        assert res["renamed"] == [{"from": "x.jpg", "to": "x_2.jpg"}]
        assert res["boxes_by_class"] == {"person": 1}  # boxes of duplicates are not imported
        names = [i["file_name"] for i in c.get(f"/api/projects/{pid}/images").json()["images"]]
        assert names == ["f_1.jpg", "x.jpg", "x_2.jpg", "y.jpg"]
        # the old image got no boxes from the duplicate
        f1 = c.get(f"/api/projects/{pid}/images").json()["images"][0]
        assert c.get(f"/api/images/{f1['id']}/annotations").json()["annotations"] == []


def test_folder_upload_saves_relative_paths(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        parts = [
            ("files", ("site/train/images/f_1.jpg", image_bytes(), "image/jpeg")),
            ("files", ("site/valid/images/f_2.jpg", image_bytes(), "image/jpeg")),
            ("files", ("site/notes.docx", b"doc", "application/octet-stream")),
            ("files", ("site/old.zip", _zip({"z.jpg": image_bytes()}), "application/zip")),
            ("files", ("labels.json", _labels(
                {"site/valid/images/f_2.jpg": [{"class_name": "rb", "bbox": [3, 3, 4, 4]}]},
                "coco"), "application/json")),
        ]
        r = c.post(f"/api/projects/{pid}/import", files=parts)
        assert r.status_code == 202, r.text
        job = r.json()
        assert job["config"] == {"upload": "files", "count": 5}
        assert job["total"] == 2
        files = _imports_dir(c) / str(job["id"]) / "files"
        assert (files / "site" / "valid" / "images" / "f_2.jpg").is_file()

        res = _run(c, job)["result"]
        assert res["format"] == "coco" and res["images_added"] == 2
        assert res["boxes_by_class"] == {"rb": 1} and res["classes_created"] == ["rb"]
        assert res["skipped"] == [{"file_name": "site/notes.docx", "reason": "not an image"},
                                  {"file_name": "site/old.zip", "reason": "not an image"}]
        assert not (_imports_dir(c) / str(job["id"])).exists()


def test_import_with_no_images_still_runs_and_cleans_up(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        job = _post_zip(c, pid, {"README.txt": b"x"}).json()
        assert (job["status"], job["total"]) == ("queued", 0)
        done = _run(c, job)
        assert done["status"] == "done" and done["progress"] == 1.0
        assert done["result"]["format"] == "images" and done["result"]["images_added"] == 0
        assert done["result"]["boxes_by_class"] == {} and done["result"]["classes_created"] == []
        assert not (_imports_dir(c) / str(job["id"])).exists()


def test_file_that_is_not_really_an_image_is_skipped(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        res = _run(c, _post_zip(c, pid, {"a.jpg": b"not a jpeg", "b.jpg": image_bytes()}).json())
        assert res["result"]["images_added"] == 1
        assert res["result"]["skipped"] == [{"file_name": "a.jpg", "reason": "not an image"}]
        assert (res["done"], res["total"]) == (2, 2)


def test_detect_job_keeps_imported_boxes(tmp_path):
    with _client(tmp_path) as c:
        s = _crane_project(c)
        entries = {"f.jpg": image_bytes(),
                   "labels.json": _labels({"f.jpg": [{"class_name": "sb", "bbox": [500, 500, 9, 9]}]})}
        _run(c, _post_zip(c, s["pid"], entries).json())
        run_job(c, s["pid"])
        iid = c.get(f"/api/projects/{s['pid']}/images").json()["images"][0]["id"]
        anns = c.get(f"/api/images/{iid}/annotations").json()["annotations"]
        assert [a["source"] for a in anns].count("import") == 1
        run_job(c, s["pid"])  # a second detect job must not delete it either
        anns = c.get(f"/api/images/{iid}/annotations").json()["annotations"]
        assert [a["source"] for a in anns].count("import") == 1


def test_unexpected_error_fails_job_keeps_rows_and_cleans_up(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        job = _post_zip(c, pid, {"a.jpg": image_bytes(), "b.jpg": image_bytes()}).json()
        real_plan = stub.plan_import

        def plan_with_bad_class(root):
            plan = real_plan(root)
            items = [stub.ImportItem(plan.items[0].rel_path, []),
                     stub.ImportItem(plan.items[1].rel_path,
                                     [{"class_name": "ghost", "bbox": [1, 1, 2, 2]}])]
            return stub.ImportPlan("images", items, [], [])  # 'ghost' not in class_names

        mod = stub.module()
        mod.plan_import = plan_with_bad_class
        c.app.state.worker.load_importers = lambda: mod
        done = _run(c, job)
        assert done["status"] == "failed" and done["error"] and done["result"] is None
        assert [i["file_name"] for i in
                c.get(f"/api/projects/{pid}/images").json()["images"]] == ["a.jpg"]
        assert not (_imports_dir(c) / str(job["id"])).exists()


def test_startup_deletes_upload_folders_of_finished_jobs(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        queued = _post_zip(c, pid, {"a.jpg": image_bytes()}).json()
        stale = _imports_dir(c) / "999"
        stale.mkdir()
        (_imports_dir(c) / "staging_old").mkdir()
    with _client(tmp_path) as c:  # restart
        left = sorted(p.name for p in _imports_dir(c).iterdir())
        assert left == [str(queued["id"])]
        assert _run(c, queued)["status"] == "done"


# ------------------------------------------------------------------ errors

def _no_job_created(c):
    conn = db.connect(c.app.state.db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    finally:
        conn.close()


def test_unknown_project_is_404_before_file_checks(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/projects/99/import")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"


@pytest.mark.parametrize("files", [
    None,
    [("files", ("x.zip", b"not a zip", "application/zip"))],
    [("files", ("../evil.jpg", b"x", "image/jpeg"))],
    [("files", ("/abs/evil.jpg", b"x", "image/jpeg"))],
    # (python-multipart already cuts "C:\\evil.jpg" to "evil.jpg", so test the / form)
    [("files", ("C:/evil.jpg", b"x", "image/jpeg"))],
    [("files", ("a/b.jpg", b"x", "image/jpeg")), ("files", ("a/b.jpg", b"y", "image/jpeg"))],
])
def test_bad_uploads_are_400_and_create_nothing(tmp_path, files):
    with _client(tmp_path) as c:
        pid = _project(c)
        kw = {"files": files} if files else {"data": {"x": "1"}}
        r = c.post(f"/api/projects/{pid}/import", **kw)
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "bad_request"
        assert _no_job_created(c)
        d = _imports_dir(c)
        assert not d.exists() or list(d.iterdir()) == []  # staging removed


def test_unsafe_zip_entry_is_400(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        r = _post_zip(c, pid, {"../etc/passwd": b"x"})
        assert r.status_code == 400
        assert r.json() == FIXTURES["error example (400 bad_request, #22 unsafe zip path)"]
        assert _no_job_created(c)


def test_size_cap_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOLABEL_IMPORT_MAX_GB", "0.000001")  # 1000 bytes
    with _client(tmp_path) as c:
        pid = _project(c)
        big = image_bytes()
        assert len(big) > 1000
        r1 = c.post(f"/api/projects/{pid}/import",
                    files=[("files", ("a.jpg", big, "image/jpeg")),
                           ("files", ("b.jpg", b"x", "image/jpeg"))])
        r2 = _post_zip(c, pid, {"a.jpg": big})
        assert (r1.status_code, r2.status_code) == (400, 400)
        assert _no_job_created(c)


def test_zip_part_among_other_parts_is_folder_upload(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        r = c.post(f"/api/projects/{pid}/import",
                   files=[("files", ("x.zip", _zip({"z.jpg": image_bytes()}), "application/zip")),
                          ("files", ("y.jpg", image_bytes(), "image/jpeg"))])
        assert r.json()["config"] == {"upload": "files", "count": 2}


# ------------------------------------------------------------------ fixture keys

def test_import_jobs_match_fixture_keys(tmp_path):
    with _client(tmp_path) as c:
        pid = _project(c)
        entries = {"valid/images/f_1.jpg": image_bytes(), "README.roboflow.txt": b"x",
                   "labels.json": _labels(
                       {"valid/images/f_1.jpg": [{"class_name": "person", "bbox": [1, 1, 5, 5]},
                                                 {"class_name": "biax", "bbox": [2, 2, 5, 5]}]})}
        job = _post_zip(c, pid, entries).json()
        same_keys(job, FIXTURES["POST /api/projects/1/import"])
        done = _run(c, job)
        same_keys(done, FIXTURES["GET /api/jobs/8 (import done)"])
        assert set(done["result"]) == set(FIXTURES["GET /api/jobs/8 (import done)"]["result"])

        folder = c.post(f"/api/projects/{pid}/import",
                        files=[("files", ("site/a.jpg", image_bytes(), "image/jpeg")),
                               ("files", ("site/notes.docx", b"d", "text/plain")),
                               ("files", ("labels.json", _labels(
                                   {"site/a.jpg": [{"class_name": "person", "bbox": [1, 1, 5, 5]},
                                                   {"class_name": "rb", "bbox": [2, 2, 5, 5]}]},
                                   "coco"), "application/json"))]).json()
        same_keys(_run(c, folder), FIXTURES["GET /api/jobs/9 (import done, folder upload)"])
