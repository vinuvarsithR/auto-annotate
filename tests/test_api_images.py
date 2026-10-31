"""Tests for endpoints 4-6: upload, list, file."""

from pathlib import Path

from tests.test_api_helpers import image_bytes, make_client, run_job, setup_project


def _project(c):
    return c.post("/api/projects", json={"name": "p"}).json()["id"]


def test_upload_jpg_and_png(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = _project(c)
        files = [("files", ("a.jpg", image_bytes("JPEG", (640, 480)), "image/jpeg")),
                 ("files", ("b.png", image_bytes("PNG", (32, 16)), "image/png"))]
        r = c.post(f"/api/projects/{pid}/images", files=files)
        assert r.status_code == 201
        body = r.json()
        assert body["skipped"] == []
        a, b = body["images"]
        assert (a["file_name"], a["width"], a["height"]) == ("a.jpg", 640, 480)
        assert (b["width"], b["height"]) == (32, 16)
        assert a["url"] == f"/api/images/{a['id']}/file"
        stored = tmp_path / "data" / "uploads" / str(pid) / f"{a['id']}_a.jpg"
        assert stored.is_file()

        f = c.get(a["url"])
        assert f.status_code == 200
        assert f.headers["content-type"] == "image/jpeg"
        assert f.content == stored.read_bytes()
        assert c.get(b["url"]).headers["content-type"] == "image/png"


def test_upload_skips_duplicates_and_non_images(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = _project(c)
        c.post(f"/api/projects/{pid}/images",
               files=[("files", ("a.jpg", image_bytes(), "image/jpeg"))])
        files = [("files", ("a.jpg", image_bytes(), "image/jpeg")),
                 ("files", ("notes.txt", b"hello", "text/plain")),
                 ("files", ("fake.jpg", b"not really a jpeg", "image/jpeg")),
                 ("files", ("png_named.jpg", image_bytes("PNG"), "image/jpeg"))]
        body = c.post(f"/api/projects/{pid}/images", files=files).json()
        assert body["images"] == []
        assert body["skipped"] == [
            {"file_name": "a.jpg", "reason": "duplicate"},
            {"file_name": "notes.txt", "reason": "not an image"},
            {"file_name": "fake.jpg", "reason": "not an image"},
            {"file_name": "png_named.jpg", "reason": "not an image"},
        ]


def test_upload_path_traversal_is_flattened(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = _project(c)
        files = [("files", ("../../evil.jpg", image_bytes(), "image/jpeg")),
                 ("files", ("..\\..\\evil2.jpg", image_bytes(), "image/jpeg"))]
        imgs = c.post(f"/api/projects/{pid}/images", files=files).json()["images"]
        assert [i["file_name"] for i in imgs] == ["evil.jpg", "evil2.jpg"]
        assert not (tmp_path / "evil.jpg").exists()
        for i in imgs:
            assert (tmp_path / "data" / "uploads" / str(pid) / f"{i['id']}_{i['file_name']}").is_file()


def test_upload_to_missing_project_404(tmp_path):
    with make_client(tmp_path)[0] as c:
        r = c.post("/api/projects/5/images",
                   files=[("files", ("a.jpg", image_bytes(), "image/jpeg"))])
        assert r.status_code == 404


def test_list_paging_and_status_filter(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = setup_project(c, n_images=3)
        pid, ids = s["pid"], s["image_ids"]
        r = c.get(f"/api/projects/{pid}/images", params={"offset": 1, "limit": 1}).json()
        assert (r["total"], r["offset"], r["limit"]) == (3, 1, 1)
        assert [i["id"] for i in r["images"]] == [ids[1]]

        run_job(c, pid, image_ids=[ids[0], ids[2]])
        r = c.get(f"/api/projects/{pid}/images", params={"status": "needs_review"}).json()
        assert r["total"] == 2
        assert [i["id"] for i in r["images"]] == [ids[0], ids[2]]
        assert r["images"][0]["counts"]["needs_review"] == 1
        assert r["images"][0]["counts"]["auto_accepted"] == 1

        assert c.get(f"/api/projects/{pid}/images", params={"status": "bogus"}).status_code == 400


def test_file_missing_image_404(tmp_path):
    with make_client(tmp_path)[0] as c:
        assert c.get("/api/images/99/file").status_code == 404
