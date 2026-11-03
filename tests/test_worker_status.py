"""Tests for the worker's status rules (docs/CONTRACT.md section 1)."""

from tests.test_api_helpers import FakePipeline, make_client, run_job, setup_project


def _anns(c, iid):
    return c.get(f"/api/images/{iid}/annotations").json()["annotations"]


def test_accept_score_decides_status(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        run_job(c, s["pid"])
        anns = _anns(c, s["image_ids"][0])
        by = {a["class_name"]: a for a in anns}
        # person: 0.9 >= accept 0.5 -> auto_accepted; rb: accept_score NULL -> needs_review
        assert by["person"]["status"] == "auto_accepted"
        assert by["rb"]["status"] == "needs_review"
        assert by["person"]["source"] == "fake" and by["person"]["job_id"] is not None
        assert by["person"]["bbox"] == [10.0, 20.0, 30.0, 40.0]
        assert by["person"]["score"] == 0.9


def test_score_equal_to_accept_is_auto_accepted(tmp_path):
    fake = FakePipeline(extra=[("person", (1, 1, 5, 5), 0.5, "fake")])
    with make_client(tmp_path, fake)[0] as c:
        s = setup_project(c, n_images=1)
        run_job(c, s["pid"])
        scores = {a["score"]: a["status"] for a in _anns(c, s["image_ids"][0])}
        assert scores[0.5] == "auto_accepted"


def test_below_min_score_dropped_and_mask_stored(tmp_path):
    mask = {"size": [720, 1280], "counts": "abc"}
    fake = FakePipeline(extra=[("sb", (1, 1, 5, 5), 0.01, "fake"),
                               ("biax", (2, 2, 6, 6), 0.2, "gdino", mask)])
    with make_client(tmp_path, fake)[0] as c:
        s = setup_project(c, n_images=1)
        run_job(c, s["pid"])
        anns = _anns(c, s["image_ids"][0])
        assert sorted(a["class_name"] for a in anns) == ["biax", "person", "rb"]
        biax = next(a for a in anns if a["class_name"] == "biax")
        assert biax["mask"] == mask and biax["source"] == "gdino"


def test_unknown_class_fails_job(tmp_path):
    fake = FakePipeline(extra=[("crane", (1, 1, 5, 5), 0.5, "fake")])
    with make_client(tmp_path, fake)[0] as c:
        s = setup_project(c, n_images=1)
        job = run_job(c, s["pid"])
        assert job["status"] == "failed" and "crane" in job["error"]
        assert _anns(c, s["image_ids"][0]) == []  # nothing half-written


def test_rerun_replaces_only_unreviewed_model_boxes(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=1)
        iid = s["image_ids"][0]
        run_job(c, s["pid"])
        first = _anns(c, iid)
        person = next(a for a in first if a["class_name"] == "person")
        rb = next(a for a in first if a["class_name"] == "rb")
        c.patch(f"/api/annotations/{person['id']}", json={"status": "accepted", "reviewer": "d"})
        human = c.post(f"/api/images/{iid}/annotations",
                       json={"class_id": s["class_ids"]["sb"], "bbox": [1, 2, 3, 4]}).json()

        run_job(c, s["pid"])
        after = _anns(c, iid)
        ids = {a["id"] for a in after}
        assert person["id"] in ids and human["id"] in ids  # reviewed + human kept
        assert rb["id"] not in ids                          # old needs_review replaced
        statuses = sorted(a["status"] for a in after)
        # the new person box sits on the accepted one (IoU 1.0), so #9 drops it
        assert statuses == ["accepted", "accepted", "needs_review"]


def test_failed_job_leaves_unreached_images_alone(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=2)
        run_job(c, s["pid"])
        before = {iid: _anns(c, iid) for iid in s["image_ids"]}

    client, _ = make_client(tmp_path, FakePipeline(fail_after=1))
    with client as c:
        job = run_job(c, s["pid"])
        assert job["status"] == "failed"
        first, second = s["image_ids"]
        assert _anns(c, second) == before[second]       # not reached -> unchanged
        assert {a["id"] for a in _anns(c, first)} != {a["id"] for a in before[first]}
