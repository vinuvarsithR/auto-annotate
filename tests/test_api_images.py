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
        same = image_bytes(colour=(1, 2, 3))
        c.post(f"/api/projects/{pid}/images",
               files=[("files", ("a.jpg", same, "image/jpeg"))])
        files = [("files", ("a.jpg", same, "image/jpeg")),
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
        assert body["renamed"] == []


# --- #4 duplicate rules: by content (sha256), not by name ---------------------

def _up(c, pid, *named):
    files = [("files", (n, b, "image/png" if n.endswith(".png") else "image/jpeg"))
             for n, b in named]
    r = c.post(f"/api/projects/{pid}/images", files=files)
    assert r.status_code == 201
    return r.json()


def test_same_bytes_other_name_is_skipped(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = _project(c)
        data = image_bytes(colour=(9, 9, 9))
        _up(c, pid, ("a.jpg", data))
        body = _up(c, pid, ("copy_of_a.jpg", data))
        assert body == {"images": [], "renamed": [],
                        "skipped": [{"file_name": "copy_of_a.jpg", "reason": "duplicate"}]}
        assert c.get(f"/api/projects/{pid}/images").json()["total"] == 1


def test_same_bytes_twice_in_one_request(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = _project(c)
        data = image_bytes(colour=(5, 6, 7))
        body = _up(c, pid, ("x.jpg", data), ("y.jpg", data), ("x.jpg", data))
        assert [i["file_name"] for i in body["images"]] == ["x.jpg"]
        assert body["skipped"] == [{"file_name": "y.jpg", "reason": "duplicate"},
                                   {"file_name": "x.jpg", "reason": "duplicate"}]
        assert body["renamed"] == []


def test_same_name_different_bytes_in_one_request(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = _project(c)
        a1, a2, a3 = (image_bytes(colour=(k, 0, 0)) for k in (10, 20, 30))
        body = _up(c, pid, ("frame_0001.jpg", a1), ("frame_0001.jpg", a2),
                   ("frame_0001.jpg", a3))
        assert [i["file_name"] for i in body["images"]] ==             ["frame_0001.jpg", "frame_0001_2.jpg", "frame_0001_3.jpg"]
        assert body["renamed"] == [{"from": "frame_0001.jpg", "to": "frame_0001_2.jpg"},
                                   {"from": "frame_0001.jpg", "to": "frame_0001_3.jpg"}]
        assert body["skipped"] == []
        for img, data in zip(body["images"], (a1, a2, a3)):  # each file kept its own bytes
            assert c.get(img["url"]).content == data
            stored = tmp_path / "data" / "uploads" / str(pid) / f"{img['id']}_{img['file_name']}"
            assert stored.read_bytes() == data


def test_same_name_different_bytes_vs_existing_and_taken_suffix(tmp_path):
    with make_client(tmp_path)[0] as c:
        pid = _project(c)
        _up(c, pid, ("a.png", image_bytes("PNG", colour=(1, 1, 1))),
            ("a_2.png", image_bytes("PNG", colour=(2, 2, 2))))
        body = _up(c, pid, ("a.png", image_bytes("PNG", colour=(3, 3, 3))))
        assert [i["file_name"] for i in body["images"]] == ["a_3.png"]
        assert body["renamed"] == [{"from": "a.png", "to": "a_3.png"}]
        # a rename keeps the extension and the real image type
        assert c.get(body["images"][0]["url"]).headers["content-type"] == "image/png"


def test_sha256_stored_but_not_in_image_json(tmp_path):
    import hashlib
    from app.db import db
    client, _ = make_client(tmp_path)
    with client as c:
        pid = _project(c)
        data = image_bytes(colour=(4, 4, 4))
        img = _up(c, pid, ("a.jpg", data))["images"][0]
        assert "sha256" not in img
        assert "sha256" not in c.get(f"/api/projects/{pid}/images").json()["images"][0]
        conn = db.connect(tmp_path / "data" / "app.db")
        assert conn.execute("SELECT sha256 FROM images WHERE id=?", (img["id"],)).fetchone()[0]             == hashlib.sha256(data).hexdigest()
        conn.close()


def test_other_project_does_not_count_as_duplicate(tmp_path):
    with make_client(tmp_path)[0] as c:
        p1 = _project(c)
        p2 = c.post("/api/projects", json={"name": "q"}).json()["id"]
        data = image_bytes(colour=(8, 8, 8))
        _up(c, p1, ("a.jpg", data))
        body = _up(c, p2, ("a.jpg", data))
        assert [i["file_name"] for i in body["images"]] == ["a.jpg"]
        assert body["skipped"] == [] and body["renamed"] == []


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
