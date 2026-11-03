"""#9: a detect job drops model boxes that overlap an existing non-rejected box of the same
class (IoU >= 0.5). FakePipeline gives each image person (10, 20, 30, 40) and rb (50, 60, 70, 80)."""

from tests import test_api_import_stub as stub
from tests.test_api_helpers import CRANE_CLASSES, image_bytes, make_client, run_job
from tests.test_api_import import _labels, _post_zip, _run


def _setup(tmp_path, boxes):
    """Project with crane classes and one image imported with `boxes`. Returns (client, pid, iid)."""
    c, _ = make_client(tmp_path)
    c.app.state.worker.load_importers = stub.module
    c.__enter__()
    pid = c.post("/api/projects", json={"name": "p"}).json()["id"]
    c.put(f"/api/projects/{pid}/classes", json={"classes": CRANE_CLASSES})
    job = _post_zip(c, pid, {"f.jpg": image_bytes(), "labels.json": _labels({"f.jpg": boxes})})
    assert _run(c, job.json())["status"] == "done"
    iid = c.get(f"/api/projects/{pid}/images").json()["images"][0]["id"]
    return c, pid, iid


def _boxes(c, iid):
    anns = c.get(f"/api/images/{iid}/annotations").json()["annotations"]
    return sorted((a["source"], a["class_name"], a["status"]) for a in anns)


def test_imported_box_blocks_overlapping_model_box_of_same_class(tmp_path):
    c, pid, iid = _setup(tmp_path, [{"class_name": "person", "bbox": [11, 21, 30, 40]}])
    try:
        assert run_job(c, pid)["status"] == "done"
        assert _boxes(c, iid) == [("fake", "rb", "needs_review"),
                                  ("import", "person", "needs_review")]
    finally:
        c.__exit__(None, None, None)


def test_overlapping_box_of_different_class_does_not_block(tmp_path):
    c, pid, iid = _setup(tmp_path, [{"class_name": "sb", "bbox": [10, 20, 30, 40]}])
    try:
        run_job(c, pid)
        assert _boxes(c, iid) == [("fake", "person", "auto_accepted"),
                                  ("fake", "rb", "needs_review"),
                                  ("import", "sb", "needs_review")]
    finally:
        c.__exit__(None, None, None)


def test_rejected_existing_box_does_not_block(tmp_path):
    c, pid, iid = _setup(tmp_path, [{"class_name": "person", "bbox": [10, 20, 30, 40]}])
    try:
        aid = c.get(f"/api/images/{iid}/annotations").json()["annotations"][0]["id"]
        c.patch(f"/api/annotations/{aid}", json={"status": "rejected"})
        run_job(c, pid)
        assert _boxes(c, iid) == [("fake", "person", "auto_accepted"),
                                  ("fake", "rb", "needs_review"),
                                  ("import", "person", "rejected")]
    finally:
        c.__exit__(None, None, None)


def test_low_overlap_same_class_is_kept(tmp_path):
    c, pid, iid = _setup(tmp_path, [{"class_name": "person", "bbox": [30, 40, 30, 40]}])  # IoU ~0.14
    try:
        run_job(c, pid)
        assert ("fake", "person", "auto_accepted") in _boxes(c, iid)
    finally:
        c.__exit__(None, None, None)


def test_human_and_reviewed_model_boxes_block_but_pending_ones_are_replaced(tmp_path):
    c, pid, iid = _setup(tmp_path, [])
    try:
        run_job(c, pid)
        run_job(c, pid)  # pending model boxes are deleted first, so no duplicates
        assert _boxes(c, iid) == [("fake", "person", "auto_accepted"),
                                  ("fake", "rb", "needs_review")]
        anns = {a["class_name"]: a for a in
                c.get(f"/api/images/{iid}/annotations").json()["annotations"]}
        c.patch(f"/api/annotations/{anns['rb']['id']}", json={"status": "accepted"})
        c.delete(f"/api/annotations/{anns['person']['id']}")
        cid = anns["person"]["class_id"]
        c.post(f"/api/images/{iid}/annotations", json={"class_id": cid, "bbox": [10, 20, 30, 40]})
        run_job(c, pid)
        assert _boxes(c, iid) == [("fake", "rb", "accepted"), ("human", "person", "accepted")]
    finally:
        c.__exit__(None, None, None)
