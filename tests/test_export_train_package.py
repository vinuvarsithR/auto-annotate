"""#23 training package (Train on Colab). Written by the integrator from docs/CONTRACT.md.

The module is imported inside a fixture, so a missing autolabel/export/train_package.py fails each
test instead of stopping the whole pytest run at collection."""

import io
import json
import zipfile
from pathlib import Path

import pytest

from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_api_helpers import image_bytes, make_client, setup_project

ROOT = Path(__file__).parents[1]
CLASSES = [{"id": 1, "name": "person"}, {"id": 2, "name": "rb"},
           {"id": 3, "name": "sb"}, {"id": 4, "name": "biax"}]
TRIANGLE = [[0.5, 0.25], [0.75, 0.25], [0.625, 0.5]]


@pytest.fixture
def tp():
    import autolabel.export as export
    return export


def img(iid, seq, frame, review="done"):
    return {"id": iid, "project_id": 1, "file_name": f"{seq}_{frame:05d}.jpg", "width": 1280,
            "height": 720, "review_status": review, "url": f"/api/images/{iid}/file"}


def ann(aid, iid, cid, status="accepted", polygon=None, bbox=(100, 100, 200, 100)):
    return {"id": aid, "image_id": iid, "class_id": cid, "class_name": CLASSES[cid - 1]["name"],
            "bbox": list(bbox), "mask": None, "polygon": polygon, "score": None,
            "source": "human", "status": status}


def seqs(spec):
    """spec: [(sequence name, frames, class_id or None)] -> images, annotations (one box per image)."""
    images, anns, iid = [], [], 0
    for seq, n, cid in spec:
        for f in range(n):
            iid += 1
            images.append(img(iid, seq, f))
            if cid:
                anns.append(ann(iid, iid, cid))
    return images, anns


def build(tp, images, anns, task="detect", classes=CLASSES, calls=None):
    def read(im):
        if calls is not None:
            calls.append(im["id"])
        return b"bytes-of-%d" % im["id"]
    blob = tp.build_train_package("eot crane", task, images, classes, anns, read)
    return zipfile.ZipFile(io.BytesIO(blob))


def split_of(z):
    return json.loads(z.read("split.json"))


def sequences_in(z, part):
    from autolabel.sequences import sequence_of
    return {sequence_of(n.rsplit("/", 1)[1]) for n in z.namelist()
            if n.startswith(f"images/{part}/")}


# ------------------------------------------------------------------ layout and data.yaml

def test_layout_and_data_yaml(tp):
    images, anns = seqs([("seqA", 10, 1), ("seqB", 3, 2), ("seqC", 2, 1)])
    z = build(tp, images, anns)
    names = set(z.namelist())
    assert {"data.yaml", "split.json", "train.ipynb"} <= names
    for n in names - {"data.yaml", "split.json", "train.ipynb"}:
        assert n.split("/")[0] in ("images", "labels") and n.split("/")[1] in ("train", "val"), n
    assert z.read("data.yaml").decode() == FIXTURES["POST /api/projects/1/train-package (zip data.yaml)"]
    # every image has its label file in the same split
    for n in names:
        if n.startswith("images/"):
            _, part, fname = n.split("/")
            assert f"labels/{part}/{Path(fname).stem}.txt" in names
    assert z.read("images/train/seqA_00000.jpg") == b"bytes-of-1"


def test_split_json_matches_fixture_keys(tp):
    images, anns = seqs([("seqA", 10, 1), ("seqB", 3, 2), ("seqC", 2, 1)])
    same_keys(split_of(build(tp, images, anns)),
              FIXTURES["POST /api/projects/1/train-package (zip split.json)"])


# ------------------------------------------------------------------ split rule

def test_smallest_sequences_first_up_to_target(tp):
    # N = 15, T = 3. Order: seqC (2), seqB (3), seqA (10). seqC fits (2 <= 3); seqB would make 5.
    images, anns = seqs([("seqA", 10, 1), ("seqB", 3, 2), ("seqC", 2, 1)])
    z = build(tp, images, anns)
    s = split_of(z)
    assert s["rule"] == "by_sequence" and s["val_target"] == 0.2 and s["val_is_train"] is False
    assert s["val"] == {"sequences": ["seqC"], "images": 2}
    assert s["train"] == {"sequences": ["seqA", "seqB"], "images": 13}
    assert sequences_in(z, "val") == {"seqC"} and sequences_in(z, "train") == {"seqA", "seqB"}
    assert s["classes_missing_in_train"] == [] and s["classes_missing_in_val"] == ["rb"]


def test_no_sequence_is_ever_split(tp):
    images, anns = seqs([(f"vid{i}", 3 + i, 1 + i % 4) for i in range(8)])
    z = build(tp, images, anns)
    assert not sequences_in(z, "train") & sequences_in(z, "val")
    s = split_of(z)
    assert s["train"]["images"] + s["val"]["images"] == len(images)


def test_class_guard_keeps_a_class_in_train(tp):
    # sb (3) lives only in seqC, so seqC may not go to val; seqB (3 <= T=3) goes instead.
    images, anns = seqs([("seqA", 10, 1), ("seqB", 3, 1), ("seqC", 2, 3)])
    s = split_of(build(tp, images, anns))
    assert s["val"]["sequences"] == ["seqB"]
    assert s["classes_missing_in_train"] == []
    assert s["classes_missing_in_val"] == ["sb"]


def test_nothing_fits_target_takes_smallest_allowed(tp):
    # N = 18, T = 4: no sequence fits, so the smallest that keeps the classes (seqB) goes alone.
    images, anns = seqs([("seqA", 10, 1), ("seqB", 8, 1)])
    s = split_of(build(tp, images, anns))
    assert s["val"] == {"sequences": ["seqB"], "images": 8}


def test_guard_impossible_takes_smallest_and_reports(tp):
    images, anns = seqs([("seqA", 10, 1), ("seqB", 8, 2)])
    s = split_of(build(tp, images, anns))
    assert s["val"]["sequences"] == ["seqB"]
    assert s["classes_missing_in_train"] == ["rb"]
    assert s["classes_missing_in_val"] == ["person"]


def test_one_sequence_val_is_train(tp):
    images, anns = seqs([("seqA", 6, 1)])
    z = build(tp, images, anns)
    s = split_of(z)
    assert s["val_is_train"] is True
    assert s["val"] == {"sequences": [], "images": 0}
    assert s["train"] == {"sequences": ["seqA"], "images": 6}
    assert s["classes_missing_in_val"] == []
    assert not [n for n in z.namelist() if n.startswith(("images/val/", "labels/val/"))]
    assert "val: images/train\n" in z.read("data.yaml").decode()


def test_split_ignores_todo_images(tp):
    images, anns = seqs([("seqA", 10, 1), ("seqB", 3, 2), ("seqC", 2, 1)])
    images += [img(100, "seqD", 0, review="todo")]
    z = build(tp, images, anns)
    assert "seqD" not in sequences_in(z, "train") | sequences_in(z, "val")


# ------------------------------------------------------------------ images and labels

def test_only_reviewed_images_and_approved_boxes(tp):
    images = [img(1, "seqA", 0), img(2, "seqA", 1, "empty"), img(3, "seqA", 2, "todo")]
    anns = [ann(1, 1, 1, "accepted"), ann(2, 1, 2, "auto_accepted"), ann(3, 1, 3, "edited"),
            ann(4, 1, 4, "needs_review"), ann(5, 1, 1, "rejected"), ann(6, 3, 1, "accepted")]
    calls = []
    z = build(tp, images, anns, calls=calls)
    assert sorted(calls) == [1, 2]
    lines = z.read("labels/train/seqA_00000.txt").decode().splitlines()
    assert sorted(int(l.split()[0]) for l in lines) == [0, 1, 2]
    assert z.read("labels/train/seqA_00001.txt") == b""
    assert "images/train/seqA_00002.jpg" not in z.namelist()


def test_detect_writes_box_lines_even_with_polygons(tp):
    images = [img(1, "seqA", 0)]
    z = build(tp, images, [ann(1, 1, 1, polygon=TRIANGLE)], task="detect")
    (line,) = z.read("labels/train/seqA_00000.txt").decode().splitlines()
    assert len(line.split()) == 5


def test_segment_polygon_and_four_corner_rows(tp):
    images = [img(1, "seqA", 0)]
    anns = [ann(1, 1, 1, polygon=TRIANGLE), ann(2, 1, 2, bbox=(128, 72, 256, 144))]
    z = build(tp, images, anns, task="segment")
    lines = z.read("labels/train/seqA_00000.txt").decode().splitlines()
    by_cls = {l.split()[0]: [float(v) for v in l.split()[1:]] for l in lines}
    assert by_cls["0"] == pytest.approx([v for pt in TRIANGLE for v in pt], abs=1e-6)
    assert by_cls["1"] == pytest.approx([0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3], abs=1e-6)


def test_segment_without_polygon_raises(tp):
    images, anns = seqs([("seqA", 3, 1)])
    with pytest.raises(tp.NoPolygons):
        build(tp, images, anns, task="segment")


def test_segment_rejected_polygon_does_not_count(tp):
    images = [img(1, "seqA", 0)]
    with pytest.raises(tp.NoPolygons):
        build(tp, images, [ann(1, 1, 1, status="rejected", polygon=TRIANGLE)], task="segment")


def test_bad_task_and_nothing_to_export(tp):
    images, anns = seqs([("seqA", 3, 1)])
    assert tuple(tp.TRAIN_TASKS) == ("detect", "segment")
    with pytest.raises(ValueError):
        build(tp, images, anns, task="pose")
    with pytest.raises(tp.NothingToExport):
        build(tp, [img(1, "seqA", 0, "todo")], [])


# ------------------------------------------------------------------ notebook

def _code(cell):
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.mark.parametrize("task,weights", [("detect", "yolo11s.pt"), ("segment", "yolo11s-seg.pt")])
def test_notebook_cells(tp, task, weights):
    images, anns = seqs([("seqA", 10, 1), ("seqB", 3, 2), ("seqC", 2, 1)])
    anns[0]["polygon"] = TRIANGLE
    nb = json.loads(build(tp, images, anns, task=task).read("train.ipynb"))
    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    assert nb["metadata"]["accelerator"] == "GPU"
    cells = nb["cells"]
    assert [c["cell_type"] for c in cells] == ["markdown", "code", "code", "code"]
    for c in cells[1:]:
        assert c["outputs"] == [] and c["execution_count"] is None
    run = f"eot_crane_{task}"

    intro = _code(cells[0])
    assert "eot crane" in intro or "eot_crane" in intro
    assert task in intro and "GPU" in intro

    pin = next(l for l in (ROOT / "requirements.txt").read_text().splitlines()
               if l.startswith("ultralytics=="))
    upload = _code(cells[1])
    assert f"!pip install -q {pin}" in upload
    assert "from google.colab import files" in upload and "files.upload()" in upload
    assert "/content/dataset" in upload

    train = _code(cells[2])
    assert f'YOLO("{weights}")' in train
    for part in ('data="/content/dataset/data.yaml"', "imgsz=960", "epochs=100",
                 'project="/content/runs"', f'name="{run}"', "exist_ok=True"):
        assert part in train, part

    download = _code(cells[3])
    assert f"/content/runs/{run}/weights/best.pt" in download
    assert f'files.download("/content/{run}_best.pt")' in download


# ------------------------------------------------------------------ route #23

def _client(tmp_path):
    return make_client(tmp_path)[0]


def _done_project(c, n=2):
    s = setup_project(c, n_images=n)
    for iid in s["image_ids"]:
        assert c.patch(f"/api/images/{iid}", json={"review_status": "done"}).status_code == 200
    return s


def test_route_returns_zip(tmp_path):
    with _client(tmp_path) as c:
        s = _done_project(c)
        c.post(f"/api/images/{s['image_ids'][0]}/annotations",
               json={"class_id": s["class_ids"]["rb"], "polygon": TRIANGLE})
        r = c.post(f"/api/projects/{s['pid']}/train-package",
                   json=FIXTURES["POST /api/projects/1/train-package (request)"])
        assert r.status_code == 200, r.text
        assert r.headers["content-type"] == "application/zip"
        assert "eot_crane_train_segment.zip" in r.headers["content-disposition"]
        z = zipfile.ZipFile(io.BytesIO(r.content))
        assert {"data.yaml", "split.json", "train.ipynb"} <= set(z.namelist())
        assert len([n for n in z.namelist() if n.startswith("images/")]) == 2


def test_route_default_task_is_detect(tmp_path):
    with _client(tmp_path) as c:
        s = _done_project(c)
        r = c.post(f"/api/projects/{s['pid']}/train-package")
        assert r.status_code == 200, r.text
        assert "eot_crane_train_detect.zip" in r.headers["content-disposition"]


@pytest.mark.parametrize("body", [{"task": "pose"}, {"task": None}, {"task": "detect", "x": 1}])
def test_route_bad_body_is_400(tmp_path, body):
    with _client(tmp_path) as c:
        s = _done_project(c)
        r = c.post(f"/api/projects/{s['pid']}/train-package", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request"


def test_route_errors(tmp_path):
    with _client(tmp_path) as c:
        assert c.post("/api/projects/99/train-package").status_code == 404
        s = setup_project(c, n_images=1)
        r = c.post(f"/api/projects/{s['pid']}/train-package")
        assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
        c.patch(f"/api/images/{s['image_ids'][0]}", json={"review_status": "done"})
        r = c.post(f"/api/projects/{s['pid']}/train-package", json={"task": "segment"})
        assert r.status_code == 409 and r.json()["error"]["code"] == "conflict"
        same_keys(r.json(), FIXTURES["error example (409 conflict, #23 segment with no polygon)"])


def test_route_missing_image_file_is_404(tmp_path):
    with _client(tmp_path) as c:
        s = _done_project(c, n=1)
        from app.db import db
        conn = db.connect(c.app.state.db_path)
        try:
            path = conn.execute("SELECT path FROM images WHERE id=?",
                                (s["image_ids"][0],)).fetchone()[0]
        finally:
            conn.close()
        Path(path).unlink()
        r = c.post(f"/api/projects/{s['pid']}/train-package")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
