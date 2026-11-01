"""Tests for image review_status: #20 PATCH /images/{iid}, the #12 409, the #5 filter."""

import pytest

from tests.test_api_helpers import image_bytes, make_client, run_job, setup_project


def _boxes(c, iid):
    return c.get(f"/api/images/{iid}/annotations").json()["annotations"]


def _reject_all(c, iid):
    for a in _boxes(c, iid):
        c.patch(f"/api/annotations/{a['id']}", json={"status": "rejected"})


def test_new_image_defaults_to_todo_everywhere(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = c.post("/api/projects", json={"name": "p"}).json()["id"]
        up = c.post(f"/api/projects/{pid}/images",
                    files=[("files", ("a.jpg", image_bytes(), "image/jpeg"))]).json()
        img = up["images"][0]
        assert img["review_status"] == "todo"                                        # #4
        assert c.get(f"/api/projects/{pid}/images").json()["images"][0]["review_status"] == "todo"  # #5
        assert c.get(f"/api/images/{img['id']}/annotations").json()["image"]["review_status"] == "todo"  # #11


@pytest.mark.parametrize("value", ["done", "empty", "todo"])
def test_patch_each_allowed_value(tmp_path, value):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)  # no boxes yet, so empty is allowed
        iid = s["image_ids"][0]
        r = c.patch(f"/api/images/{iid}", json={"review_status": value, "reviewer": "dhyan"})
        assert r.status_code == 200
        img = r.json()
        assert img["id"] == iid and img["review_status"] == value
        assert c.get(f"/api/images/{iid}/annotations").json()["image"]["review_status"] == value
        # same value again: no-op 200
        assert c.patch(f"/api/images/{iid}", json={"review_status": value}).json() == img


def test_patch_any_order_and_done_with_needs_review(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        run_job(c, s["pid"])  # 1 auto_accepted + 1 needs_review box
        iid = s["image_ids"][0]
        img = c.patch(f"/api/images/{iid}", json={"review_status": "done"}).json()
        assert img["review_status"] == "done" and img["counts"]["needs_review"] == 1
        assert c.patch(f"/api/images/{iid}", json={"review_status": "todo"}).json()["review_status"] == "todo"
        _reject_all(c, iid)
        assert c.patch(f"/api/images/{iid}", json={"review_status": "empty"}).json()["review_status"] == "empty"
        assert c.patch(f"/api/images/{iid}", json={"review_status": "done"}).json()["review_status"] == "done"


@pytest.mark.parametrize("body", [{"review_status": "finished"}, {"review_status": "TODO"},
                                  {"review_status": ""}])
def test_patch_bad_value_400(tmp_path, body):
    with make_client(tmp_path)[0] as c:
        iid = setup_project(c, n_images=1)["image_ids"][0]
        r = c.patch(f"/api/images/{iid}", json=body)
        assert r.status_code == 400
        assert r.json() == {"error": {"code": "bad_request",
                                      "message": "review_status must be one of todo, done, empty"}}


@pytest.mark.parametrize("body", [{}, {"reviewer": "dhyan"}, {"review_status": None},
                                  {"review_status": 3}, [1, 2]])
def test_patch_missing_or_bad_body_400(tmp_path, body):
    with make_client(tmp_path)[0] as c:
        iid = setup_project(c, n_images=1)["image_ids"][0]
        r = c.patch(f"/api/images/{iid}", json=body)
        assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_request")
        assert c.get(f"/api/images/{iid}/annotations").json()["image"]["review_status"] == "todo"


def test_patch_unknown_image_404(tmp_path):
    with make_client(tmp_path)[0] as c:
        r = c.patch("/api/images/999", json={"review_status": "done"})
        assert r.status_code == 404
        assert r.json() == {"error": {"code": "not_found", "message": "image 999 not found"}}


def test_patch_empty_with_live_boxes_409(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        run_job(c, s["pid"])
        iid = s["image_ids"][0]
        c.patch(f"/api/images/{iid}", json={"review_status": "done"})
        r = c.patch(f"/api/images/{iid}", json={"review_status": "empty"})
        assert r.status_code == 409
        assert r.json() == {"error": {"code": "conflict",
                                      "message": f"image {iid} has 2 boxes that are not rejected"}}
        # nothing changed
        assert c.get(f"/api/images/{iid}/annotations").json()["image"]["review_status"] == "done"
        # one rejected, one left: still 409, count goes down
        first = _boxes(c, iid)[0]
        c.patch(f"/api/annotations/{first['id']}", json={"status": "rejected"})
        r = c.patch(f"/api/images/{iid}", json={"review_status": "empty"})
        assert r.status_code == 409 and "has 1 boxes" in r.json()["error"]["message"]


def test_patch_empty_blocked_by_accepted_human_box(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        iid = s["image_ids"][0]
        c.post(f"/api/images/{iid}/annotations",
               json={"class_id": s["class_ids"]["rb"], "bbox": [1, 2, 3, 4]})
        assert c.patch(f"/api/images/{iid}", json={"review_status": "empty"}).status_code == 409


def test_add_box_to_empty_image_409(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        iid = s["image_ids"][0]
        c.patch(f"/api/images/{iid}", json={"review_status": "empty"})
        body = {"class_id": s["class_ids"]["rb"], "bbox": [1, 2, 3, 4]}
        r = c.post(f"/api/images/{iid}/annotations", json=body)
        assert r.status_code == 409
        assert r.json() == {"error": {"code": "conflict", "message":
                                      f"image {iid} is marked empty; set review_status to todo first"}}
        assert _boxes(c, iid) == []
        # back to todo, then the box is allowed
        c.patch(f"/api/images/{iid}", json={"review_status": "todo"})
        assert c.post(f"/api/images/{iid}/annotations", json=body).status_code == 201
        # done images still take new boxes
        c.patch(f"/api/images/{iid}", json={"review_status": "done"})
        assert c.post(f"/api/images/{iid}/annotations", json=body).status_code == 201


def test_detect_job_on_empty_image_keeps_it_empty(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        iid = s["image_ids"][0]
        c.patch(f"/api/images/{iid}", json={"review_status": "empty"})
        assert run_job(c, s["pid"])["status"] == "done"
        body = c.get(f"/api/images/{iid}/annotations").json()
        assert body["image"]["review_status"] == "empty" and len(body["annotations"]) == 2


def _ids(c, pid, **params):
    body = c.get(f"/api/projects/{pid}/images", params=params).json()
    assert body["total"] == len(body["images"])
    return [i["id"] for i in body["images"]]


def test_list_filter_review_status(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=4)
        pid, (a, b, d, e) = s["pid"], s["image_ids"]
        c.patch(f"/api/images/{b}", json={"review_status": "done"})
        c.patch(f"/api/images/{d}", json={"review_status": "done"})
        c.patch(f"/api/images/{e}", json={"review_status": "empty"})
        assert _ids(c, pid, review_status="todo") == [a]
        assert _ids(c, pid, review_status="done") == [b, d]
        assert _ids(c, pid, review_status="empty") == [e]
        assert _ids(c, pid) == [a, b, d, e]
        r = c.get(f"/api/projects/{pid}/images", params={"review_status": "nope"})
        assert r.status_code == 400
        assert r.json()["error"] == {"code": "bad_request",
                                     "message": "review_status must be one of todo, done, empty"}


def test_list_filter_review_status_with_status(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=3)
        pid, (a, b, d) = s["pid"], s["image_ids"]
        run_job(c, pid)  # every image: 1 auto_accepted + 1 needs_review
        c.patch(f"/api/images/{b}", json={"review_status": "done"})
        # d: done, and its needs_review box accepted -> no needs_review left
        c.patch(f"/api/images/{d}", json={"review_status": "done"})
        for box in _boxes(c, d):
            if box["status"] == "needs_review":
                c.patch(f"/api/annotations/{box['id']}", json={"status": "accepted"})
        assert _ids(c, pid, status="needs_review") == [a, b]
        assert _ids(c, pid, status="needs_review", review_status="done") == [b]
        assert _ids(c, pid, status="needs_review", review_status="todo") == [a]
        assert _ids(c, pid, status="accepted", review_status="done") == [d]
        assert _ids(c, pid, status="accepted", review_status="todo") == []
        # paging still counts the filtered total
        body = c.get(f"/api/projects/{pid}/images",
                     params={"review_status": "done", "limit": 1}).json()
        assert body["total"] == 2 and len(body["images"]) == 1
