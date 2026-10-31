"""Tests for endpoints 11-13: list, add human box, PATCH review rules."""

from tests.test_api_helpers import make_client, run_job, setup_project


def _setup(c):
    s = setup_project(c, n_images=1)
    run_job(c, s["pid"])
    iid = s["image_ids"][0]
    anns = c.get(f"/api/images/{iid}/annotations").json()["annotations"]
    return s, iid, {a["class_name"]: a for a in anns}


def test_list_with_status_filter_and_image_counts(tmp_path):
    with make_client(tmp_path)[0] as c:
        s, iid, _ = _setup(c)
        body = c.get(f"/api/images/{iid}/annotations", params={"status": "needs_review"}).json()
        assert [a["class_name"] for a in body["annotations"]] == ["rb"]
        assert body["image"]["counts"]["auto_accepted"] == 1
        assert body["image"]["counts"]["needs_review"] == 1
        assert c.get("/api/images/99/annotations").status_code == 404


def test_add_human_box(tmp_path):
    with make_client(tmp_path)[0] as c:
        s, iid, _ = _setup(c)
        r = c.post(f"/api/images/{iid}/annotations",
                   json={"class_id": s["class_ids"]["biax"], "bbox": [100, 200, 40, 50],
                         "mask": None, "reviewer": "dhyan"})
        assert r.status_code == 201
        a = r.json()
        assert (a["source"], a["score"], a["status"], a["job_id"]) == ("human", None, "accepted", None)
        assert a["class_name"] == "biax" and a["bbox"] == [100.0, 200.0, 40.0, 50.0]
        assert a["reviewer"] == "dhyan"


def test_add_human_box_validation(tmp_path):
    with make_client(tmp_path)[0] as c:
        s, iid, _ = _setup(c)
        other = c.post("/api/projects", json={"name": "other"}).json()["id"]
        oc = c.put(f"/api/projects/{other}/classes",
                   json={"classes": [{"name": "x", "description": "x"}]}).json()["classes"][0]
        r = c.post(f"/api/images/{iid}/annotations", json={"class_id": oc["id"], "bbox": [1, 1, 1, 1]})
        assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_class_id")
        r = c.post(f"/api/images/{iid}/annotations",
                   json={"class_id": s["class_ids"]["rb"], "bbox": [1, 1, 0, 1]})
        assert r.status_code == 400
        r = c.post(f"/api/images/{iid}/annotations",
                   json={"class_id": s["class_ids"]["rb"], "bbox": [1, 1, 1]})
        assert r.status_code == 400


def test_patch_accept_and_reject(tmp_path):
    with make_client(tmp_path)[0] as c:
        _, iid, by = _setup(c)
        a = c.patch(f"/api/annotations/{by['rb']['id']}",
                    json={"status": "accepted", "reviewer": "dhyan"}).json()
        assert a["status"] == "accepted" and a["reviewer"] == "dhyan"
        r = c.patch(f"/api/annotations/{by['person']['id']}", json={"status": "rejected"}).json()
        assert r["status"] == "rejected"
        # rejected row is kept
        ids = [x["id"] for x in c.get(f"/api/images/{iid}/annotations").json()["annotations"]]
        assert by["person"]["id"] in ids


def test_patch_edit_box_or_class_gives_edited(tmp_path):
    with make_client(tmp_path)[0] as c:
        s, _, by = _setup(c)
        a = c.patch(f"/api/annotations/{by['rb']['id']}",
                    json={"bbox": [810, 298, 250, 70], "class_id": s["class_ids"]["biax"],
                          "reviewer": "dhyan"}).json()
        assert a["status"] == "edited" and a["class_name"] == "biax"
        assert a["bbox"] == [810.0, 298.0, 250.0, 70.0]
        assert a["score"] == 0.3  # model score kept
        assert a["updated_at"] >= a["created_at"]


def test_patch_explicit_status_wins_over_edit(tmp_path):
    with make_client(tmp_path)[0] as c:
        _, _, by = _setup(c)
        a = c.patch(f"/api/annotations/{by['rb']['id']}",
                    json={"bbox": [1, 2, 3, 4], "status": "accepted"}).json()
        assert a["status"] == "accepted" and a["bbox"] == [1.0, 2.0, 3.0, 4.0]
        b = c.patch(f"/api/annotations/{by['person']['id']}",
                    json={"bbox": [1, 2, 3, 4], "status": "rejected"}).json()
        assert b["status"] == "rejected"


def test_patch_undo_accept_back_to_needs_review(tmp_path):
    with make_client(tmp_path)[0] as c:
        _, _, by = _setup(c)
        aid = by["rb"]["id"]
        assert c.patch(f"/api/annotations/{aid}", json={"status": "accepted"}).json()["status"] == "accepted"
        a = c.patch(f"/api/annotations/{aid}", json={"status": "needs_review", "reviewer": "dhyan"}).json()
        assert a["status"] == "needs_review"


def test_patch_undo_move_restores_bbox_and_status(tmp_path):
    with make_client(tmp_path)[0] as c:
        s, _, by = _setup(c)
        old = by["person"]  # auto_accepted model box
        moved = c.patch(f"/api/annotations/{old['id']}",
                        json={"bbox": [810, 298, 250, 70], "class_id": s["class_ids"]["biax"]}).json()
        assert moved["status"] == "edited"
        back = c.patch(f"/api/annotations/{old['id']}",
                       json={"status": old["status"], "bbox": old["bbox"],
                             "class_id": old["class_id"], "reviewer": "dhyan"}).json()
        assert back["status"] == "auto_accepted"
        assert (back["bbox"], back["class_id"]) == (old["bbox"], old["class_id"])


def test_patch_same_bbox_is_not_an_edit(tmp_path):
    with make_client(tmp_path)[0] as c:
        _, _, by = _setup(c)
        a = c.patch(f"/api/annotations/{by['rb']['id']}",
                    json={"bbox": by["rb"]["bbox"], "status": "accepted"}).json()
        assert a["status"] == "accepted"


def test_patch_mask_and_reviewer_only(tmp_path):
    with make_client(tmp_path)[0] as c:
        _, _, by = _setup(c)
        r = c.patch(f"/api/annotations/{by['rb']['id']}", json={"reviewer": "x"}).json()
        assert r["status"] == "needs_review" and r["reviewer"] == "x"
        mask = {"size": [720, 1280], "counts": "xyz"}
        r = c.patch(f"/api/annotations/{by['rb']['id']}", json={"mask": mask}).json()
        assert r["status"] == "edited" and r["mask"] == mask


def test_patch_bad_status_and_missing(tmp_path):
    with make_client(tmp_path)[0] as c:
        _, _, by = _setup(c)
        r = c.patch(f"/api/annotations/{by['rb']['id']}", json={"status": "approved"})
        assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_request")
        for st in ("auto_accepted", "needs_review", "accepted", "rejected", "edited"):
            r = c.patch(f"/api/annotations/{by['rb']['id']}", json={"status": st})
            assert (r.status_code, r.json()["status"]) == (200, st)
        assert c.patch("/api/annotations/9999", json={"status": "accepted"}).status_code == 404


def test_delete_annotation_any_source(tmp_path):
    with make_client(tmp_path)[0] as c:
        s, iid, by = _setup(c)
        human = c.post(f"/api/images/{iid}/annotations",
                       json={"class_id": s["class_ids"]["biax"], "bbox": [1, 2, 3, 4]}).json()
        rejected = c.patch(f"/api/annotations/{by['person']['id']}", json={"status": "rejected"}).json()
        for aid in (human["id"], by["rb"]["id"], rejected["id"]):
            r = c.delete(f"/api/annotations/{aid}")
            assert r.status_code == 204 and r.content == b""
        assert c.get(f"/api/images/{iid}/annotations").json()["annotations"] == []
        r = c.delete(f"/api/annotations/{human['id']}")
        assert (r.status_code, r.json()["error"]["code"]) == (404, "not_found")
