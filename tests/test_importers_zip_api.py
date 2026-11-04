"""#22 with the real autolabel.importers (tests/test_api_import.py uses a stub): a zip with
identical repeated entries, as real Roboflow YOLO exports have, imports normally."""

import io
import warnings
import zipfile

from tests.test_api_helpers import image_bytes, make_client


def test_zip_with_identical_repeats_imports_two_images(tmp_path):
    data_yaml = "names: ['person', 'biax']\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # zipfile warns about the repeats
        zf.writestr("data.yaml", data_yaml)
        zf.writestr("train/images/a.png", image_bytes("PNG", size=(100, 50)))
        zf.writestr("data.yaml", data_yaml)
        zf.writestr("train/images/b.png", image_bytes("PNG", size=(100, 50)))
        zf.writestr("train/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
        zf.writestr("train/labels/b.txt", "1 0.25 0.5 0.5 1.0\n")
        zf.writestr("data.yaml", data_yaml)

    c, _ = make_client(tmp_path)
    with c:
        pid = c.post("/api/projects", json={"name": "eot_crane"}).json()["id"]
        res = c.post(f"/api/projects/{pid}/import",
                     files=[("files", ("roboflow.zip", buf.getvalue(), "application/zip"))])
        assert res.status_code == 202, res.text
        assert res.json()["total"] == 2

        assert c.app.state.worker.run_once()
        job = c.get(f"/api/jobs/{res.json()['id']}").json()
        assert job["status"] == "done"
        assert job["result"]["format"] == "yolo"
        assert job["result"]["images_added"] == 2
        assert job["result"]["boxes_by_class"] == {"person": 1, "biax": 1}
        assert job["result"]["skipped"] == []
        assert c.get(f"/api/projects/{pid}/images").json()["total"] == 2
