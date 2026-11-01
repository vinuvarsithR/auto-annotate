"""
Real API responses must have the same keys as docs/fixtures/api_samples.json,
at every level (Project.counts, Image.counts, Job.config, Annotation.mask, ...).
"""

import json
from pathlib import Path

from tests.test_api_helpers import FakePipeline, image_bytes, make_client, run_job, setup_project

FIXTURES = json.loads((Path(__file__).parents[1] / "docs" / "fixtures" / "api_samples.json")
                      .read_text(encoding="utf-8"))


def same_keys(real, sample, where="$"):
    """Assert the two JSON values have the same dict keys everywhere."""
    if isinstance(sample, dict) and isinstance(real, dict):
        assert set(real) == set(sample), f"{where}: {sorted(set(real) ^ set(sample))}"
        for k in sample:
            same_keys(real[k], sample[k], f"{where}.{k}")
    elif isinstance(sample, list) and isinstance(real, list):
        if sample:
            for i, item in enumerate(real):
                same_keys(item, sample[0], f"{where}[{i}]")
    # anything else (numbers, strings, None vs dict for mask): value, not shape


def test_all_endpoints_match_fixture_keys(tmp_path):
    mask = {"size": [720, 1280], "counts": "abc"}
    fake = FakePipeline(extra=[("biax", (812, 300.5, 240, 66), 0.42, "owlv2_image", mask)])
    client, _ = make_client(tmp_path, fake)
    with client as c:
        created = c.post("/api/projects", json={"name": "eot_crane"}).json()
        same_keys(created, FIXTURES["POST /api/projects"])
        pid = created["id"]

        files = [("files", (f"im{i}.jpg", image_bytes(), "image/jpeg")) for i in range(3)]
        files.append(("files", ("notes.txt", b"x", "text/plain")))
        up = c.post(f"/api/projects/{pid}/images", files=files).json()
        same_keys(up, FIXTURES["POST /api/projects/1/images"])
        iid = up["images"][0]["id"]

        from tests.test_api_helpers import CRANE_CLASSES
        same_keys(c.put(f"/api/projects/{pid}/classes", json={"classes": CRANE_CLASSES}).json(),
                  FIXTURES["PUT /api/projects/1/classes"])
        same_keys(c.get(f"/api/projects/{pid}/classes").json(),
                  FIXTURES["GET /api/projects/1/classes"])

        job = c.post(f"/api/projects/{pid}/jobs",
                     json={"model": "pipeline", "image_ids": None, "options": {}}).json()
        same_keys(job, FIXTURES["POST /api/projects/1/jobs"])
        c.app.state.worker.run_once()
        same_keys(c.get(f"/api/jobs/{job['id']}").json(), FIXTURES["GET /api/jobs/5"])

        same_keys(c.get("/api/projects").json(), FIXTURES["GET /api/projects"])
        same_keys(c.get(f"/api/projects/{pid}").json(), FIXTURES["GET /api/projects/1"])
        same_keys(c.get(f"/api/projects/{pid}/images").json(),
                  FIXTURES["GET /api/projects/1/images"])
        same_keys(c.get(f"/api/projects/{pid}/images", params={"status": "needs_review"}).json(),
                  FIXTURES["GET /api/projects/1/images?status=needs_review"])

        anns = c.get(f"/api/images/{iid}/annotations").json()
        for key in ("GET /api/images/10/annotations", "GET /api/images/12/annotations"):
            same_keys(anns, FIXTURES[key])
        masked = [a for a in anns["annotations"] if a["mask"] is not None]
        assert masked, "need at least one box with a mask"
        fixture_mask = next(a["mask"] for a in FIXTURES["all_annotations"] if a["mask"])
        same_keys(masked[0]["mask"], fixture_mask)

        cid = anns["annotations"][0]["class_id"]
        human = c.post(f"/api/images/{iid}/annotations",
                       json={"class_id": cid, "bbox": [100, 200, 40, 50], "mask": None,
                             "reviewer": "dhyan"}).json()
        same_keys(human, FIXTURES["POST /api/images/10/annotations"])

        aid = anns["annotations"][1]["id"]
        same_keys(c.patch(f"/api/annotations/{aid}",
                          json={"status": "accepted", "reviewer": "dhyan"}).json(),
                  FIXTURES["PATCH /api/annotations/102 (accept)"])
        same_keys(c.patch(f"/api/annotations/{aid}",
                          json={"bbox": [810, 298, 250, 70], "reviewer": "dhyan"}).json(),
                  FIXTURES["PATCH /api/annotations/105 (edit)"])

        same_keys(c.patch(f"/api/annotations/{aid}",
                          json={"status": "needs_review", "reviewer": "dhyan"}).json(),
                  FIXTURES["PATCH /api/annotations/102 (undo)"])

        same_keys(c.get("/api/models").json(), FIXTURES["GET /api/models"])

        # #20 and the review_status filter / errors
        bad = c.patch(f"/api/images/{iid}", json={"review_status": "finished"}).json()
        same_keys(bad, FIXTURES["error example (400 bad review_status)"])
        assert bad == FIXTURES["error example (400 bad review_status)"]
        same_keys(c.patch(f"/api/images/{iid}", json={"review_status": "empty"}).json(),
                  FIXTURES["error example (409 conflict, #20 empty with boxes)"])
        same_keys(c.patch(f"/api/images/{iid}", json={"review_status": "done", "reviewer": "dhyan"})
                  .json(), FIXTURES["PATCH /api/images/10"])
        same_keys(c.get(f"/api/projects/{pid}/images", params={"review_status": "done"}).json(),
                  FIXTURES["GET /api/projects/1/images?review_status=done"])
        empty_iid = up["images"][2]["id"]
        for a in c.get(f"/api/images/{empty_iid}/annotations").json()["annotations"]:
            c.patch(f"/api/annotations/{a['id']}", json={"status": "rejected"})
        assert c.patch(f"/api/images/{empty_iid}", json={"review_status": "empty"}).status_code == 200
        same_keys(c.post(f"/api/images/{empty_iid}/annotations",
                         json={"class_id": cid, "bbox": [1, 2, 3, 4]}).json(),
                  FIXTURES["error example (409 conflict, #12 box on empty image)"])

        same_keys(c.patch(f"/api/projects/{pid}", json={"name": "eot_crane_v2"}).json(),
                  FIXTURES["PATCH /api/projects/1"])

        same_keys(c.get("/api/images/99/annotations").json(), FIXTURES["error example (404)"])

        # 204 endpoints: the fixture value is null, the real body is empty
        assert FIXTURES["DELETE /api/annotations/109"] is None
        r = c.delete(f"/api/annotations/{human['id']}")
        assert (r.status_code, r.content) == (204, b"")

        c.post(f"/api/projects/{pid}/jobs", json={"model": "pipeline"})  # stays queued
        same_keys(c.delete(f"/api/projects/{pid}").json(), FIXTURES["error example (409 conflict)"])
        c.app.state.worker.run_once()
        assert FIXTURES["DELETE /api/projects/1"] is None
        r = c.delete(f"/api/projects/{pid}")
        assert (r.status_code, r.content) == (204, b"")


def test_failed_job_matches_fixture_keys(tmp_path):
    client, _ = make_client(tmp_path, FakePipeline(fail_after=0))
    with client as c:
        s = setup_project(c, n_images=1)
        job = run_job(c, s["pid"])
        same_keys(job, FIXTURES["GET /api/jobs/6"])
        assert job["status"] == "failed" and job["error"]
