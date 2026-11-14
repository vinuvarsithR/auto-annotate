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


# ------------------------------------------------------------------ options (#23 change, 2026-11-14)

def _cells(z):
    return [_code(c) for c in json.loads(z.read("train.ipynb"))["cells"]]


def test_options_reach_cells_3_and_4(tp):
    images, anns = seqs([("seqA", 10, 1), ("seqB", 3, 2)])
    anns[0]["polygon"] = TRIANGLE
    blob = tp.build_train_package("eot crane", "segment", images, CLASSES, anns,
                                  lambda im: b"x", epochs=50, imgsz=640,
                                  base_weights="yolo11m-seg.pt")
    cells = _cells(zipfile.ZipFile(io.BytesIO(blob)))
    assert len(cells) == 4
    assert 'YOLO("yolo11m-seg.pt")' in cells[2]
    assert "imgsz=640" in cells[2] and "epochs=50" in cells[2]
    assert "imgsz=960" not in cells[2] and "epochs=100" not in cells[2]
    assert "imgsz=640" in cells[3] and '"base_weights": "yolo11m-seg.pt"' in cells[3]
    assert '"epochs": 50' in cells[3]


def test_options_none_base_weights_is_default(tp):
    images, anns = seqs([("seqA", 3, 1)])
    blob = tp.build_train_package("p", "detect", images, CLASSES, anns, lambda im: b"x",
                                  base_weights=None)
    assert 'YOLO("yolo11s.pt")' in _cells(zipfile.ZipFile(io.BytesIO(blob)))[2]


@pytest.mark.parametrize("task,kw", [
    ("detect", {"epochs": 0}), ("detect", {"epochs": 501}), ("detect", {"epochs": True}),
    ("detect", {"epochs": 5.0}), ("detect", {"imgsz": 319}), ("detect", {"imgsz": 2080}),
    ("detect", {"imgsz": 650}), ("detect", {"imgsz": "960"}),
    ("detect", {"base_weights": "yolo11s-seg.pt"}), ("segment", {"base_weights": "yolo11s.pt"}),
    ("detect", {"base_weights": "yolov8s.pt"}), ("detect", {"base_weights": 3}),
])
def test_bad_options_raise_before_other_checks(tp, task, kw):
    # no exportable image: a bad option must still be ValueError (400), not NothingToExport (409)
    with pytest.raises(ValueError):
        tp.build_train_package("p", task, [img(1, "seqA", 0, "todo")], CLASSES, [],
                               lambda im: b"x", **kw)


def test_option_edges_are_allowed(tp):
    for task, w in (("detect", "yolo11x.pt"), ("segment", "yolo11n-seg.pt")):
        assert tp.check_train_options(task, 1, 320, w) == w
        assert tp.check_train_options(task, 500, 2048, None) == tp.train_package.BASE_WEIGHTS[task]


# ------------------------------------------------------------------ cell 4 run with fake ultralytics

class _FakeMaps:
    def __init__(self, map50, map_):
        self.map50, self.map = map50, map_


def _run_cell4(tp, tmp_path, monkeypatch, task):
    """Runs notebook cell 4 with /content -> tmp_path and fake google.colab / ultralytics."""
    import sys
    import types
    images, anns = seqs([("seqA", 10, 1), ("seqB", 3, 2), ("seqC", 2, 1)])
    anns[0]["polygon"] = TRIANGLE
    z = build(tp, images, anns, task=task)
    run = f"eot_crane_{task}"
    content = tmp_path / "content"
    (content / "dataset").mkdir(parents=True)
    (content / "dataset" / "split.json").write_bytes(z.read("split.json"))
    (content / "runs" / run / "weights").mkdir(parents=True)
    (content / "runs" / run / "weights" / "best.pt").write_bytes(b"weights")

    calls = {"val": [], "download": []}

    class FakeYOLO:
        def __init__(self, path):
            self.path = path

        def val(self, **kw):
            calls["val"].append((self.path, kw))
            return types.SimpleNamespace(box=_FakeMaps(0.81, 0.55), seg=_FakeMaps(0.78, 0.49))

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = FakeYOLO
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(download=lambda p: calls["download"].append(p))
    google = types.ModuleType("google")
    google.colab = colab
    for name, mod in (("ultralytics", ultra), ("google", google), ("google.colab", colab)):
        monkeypatch.setitem(sys.modules, name, mod)

    code = _cells(z)[3].replace("/content", content.as_posix())
    exec(compile(code, "cell4", "exec"), {})
    return z, run, content, calls


@pytest.mark.parametrize("task", ["detect", "segment"])
def test_cell4_writes_metrics_that_read_metrics_accepts(tp, tmp_path, monkeypatch, task):
    z, run, content, calls = _run_cell4(tp, tmp_path, monkeypatch, task)
    pt, js = content / f"{run}_best.pt", content / f"{run}_metrics.json"
    assert pt.read_bytes() == b"weights"
    assert calls["download"] == [pt.as_posix(), js.as_posix()]
    ((path, kw),) = calls["val"]
    assert path == pt.as_posix()
    assert kw == {"data": f"{content.as_posix()}/dataset/data.yaml", "imgsz": 960, "split": "val"}

    doc = json.loads(js.read_text(encoding="utf-8"))
    assert doc["format"] == "autolabel-metrics-1" and doc["source"] == "colab"
    assert doc["task"] == task
    assert doc["split"] == split_of(z)
    out = tp.read_metrics(js.read_bytes(), task)
    assert out["split"] == split_of(z)
    m = out["metrics"]
    assert m["box"] == {"map50": 0.81, "map50_95": 0.55}
    assert m["mask"] == ({"map50": 0.78, "map50_95": 0.49} if task == "segment" else None)
    assert m["epochs"] == 100 and m["imgsz"] == 960
    assert m["base_weights"] == tp.train_package.BASE_WEIGHTS[task]
    assert m["finished_at"].endswith("Z")
    same_keys(doc, FIXTURES["metrics file (eot_crane_segment_metrics.json)"])


def test_cell4_text(tp):
    images, anns = seqs([("seqA", 3, 1)])
    cell = _cells(build(tp, images, anns))[3]
    assert ".val(" in cell and 'split="val"' in cell and "_metrics.json" in cell
    assert cell.count("files.download(") == 2
    assert 'files.download("/content/eot_crane_detect_best.pt")' in cell
    assert 'files.download("/content/eot_crane_detect_metrics.json")' in cell
    assert "/content/dataset/split.json" in cell


# ------------------------------------------------------------------ read_metrics

def _metrics_doc():
    return json.loads(json.dumps(FIXTURES["metrics file (eot_crane_segment_metrics.json)"]))


def _enc(doc):
    return json.dumps(doc).encode()


def test_read_metrics_fixture(tp):
    doc = _metrics_doc()
    out = tp.read_metrics(_enc(doc), "segment")
    assert out == {"metrics": doc["metrics"], "split": doc["split"]}


def test_read_metrics_detect_mask_null(tp):
    doc = _metrics_doc()
    doc["task"], doc["metrics"]["mask"] = "detect", None
    assert tp.read_metrics(_enc(doc), "detect")["metrics"]["mask"] is None
    del doc["metrics"]["mask"]
    assert tp.read_metrics(_enc(doc), "detect")["metrics"]["mask"] is None


def test_read_metrics_ignores_other_keys(tp):
    doc = _metrics_doc()
    doc["extra"] = 1
    doc["metrics"]["extra"] = {"x": 1}
    doc["metrics"]["box"]["precision"] = 0.5
    out = tp.read_metrics(_enc(doc), "segment")["metrics"]
    assert "extra" not in out and out["box"] == {"map50": 0.81, "map50_95": 0.55}


_DEL = object()


def _set(path, value):
    def change(doc):
        d = doc
        for k in path[:-1]:
            d = d[k]
        if value is _DEL:
            del d[path[-1]]
        else:
            d[path[-1]] = value
    return change


BAD_RULES = {
    "format wrong": _set(("format",), "autolabel-metrics-2"),
    "format missing": _set(("format",), _DEL),
    "task mismatch": _set(("task",), "detect"),
    "split missing": _set(("split",), _DEL),
    "split not object": _set(("split",), []),
    "metrics missing": _set(("metrics",), _DEL),
    "box missing": _set(("metrics", "box"), _DEL),
    "box map50 > 1": _set(("metrics", "box", "map50"), 1.2),
    "box map50 < 0": _set(("metrics", "box", "map50"), -0.1),
    "box map50_95 str": _set(("metrics", "box", "map50_95"), "0.5"),
    "box map50 bool": _set(("metrics", "box", "map50"), True),
    "box map50 missing": _set(("metrics", "box", "map50"), _DEL),
    "mask null for segment": _set(("metrics", "mask"), None),
    "mask map50_95 > 1": _set(("metrics", "mask", "map50_95"), 2),
    "epochs 0": _set(("metrics", "epochs"), 0),
    "epochs float": _set(("metrics", "epochs"), 50.0),
    "imgsz missing": _set(("metrics", "imgsz"), _DEL),
    "imgsz negative": _set(("metrics", "imgsz"), -960),
    "base_weights not str": _set(("metrics", "base_weights"), None),
    "finished_at not iso": _set(("metrics", "finished_at"), "yesterday"),
    "finished_at no tz": _set(("metrics", "finished_at"), "2026-11-14T09:30:00"),
    "finished_at not utc": _set(("metrics", "finished_at"), "2026-11-14T09:30:00+05:30"),
    "finished_at number": _set(("metrics", "finished_at"), 1731576600),
}


@pytest.mark.parametrize("rule", list(BAD_RULES))
def test_read_metrics_bad_rule(tp, rule):
    doc = _metrics_doc()
    BAD_RULES[rule](doc)
    with pytest.raises(tp.BadMetrics) as e:
        tp.read_metrics(_enc(doc), "segment")
    assert str(e.value).startswith("metrics file: ")


def test_read_metrics_detect_with_mask_is_bad(tp):
    doc = _metrics_doc()
    doc["task"] = "detect"
    with pytest.raises(tp.BadMetrics):
        tp.read_metrics(_enc(doc), "detect")


@pytest.mark.parametrize("data", [b"", b"\xff\xfe{}", b"{not json", b"[]", b"null",
                                  b" " * (1024 * 1024) + b"{}"])
def test_read_metrics_bad_bytes(tp, data):
    with pytest.raises(tp.BadMetrics):
        tp.read_metrics(data, "segment")


def test_read_metrics_message_matches_fixture(tp):
    doc = _metrics_doc()
    doc["format"] = "x"
    with pytest.raises(tp.BadMetrics) as e:
        tp.read_metrics(_enc(doc), "segment")
    assert str(e.value) == FIXTURES["error example (400 bad_metrics, #24)"]["error"]["message"]


# ------------------------------------------------------------------ route #23 options

def test_route_options_reach_notebook(tmp_path):
    with _client(tmp_path) as c:
        s = _done_project(c)
        r = c.post(f"/api/projects/{s['pid']}/train-package",
                   json={"task": "detect", "epochs": 30, "imgsz": 1280,
                         "base_weights": "yolo11n.pt"})
        assert r.status_code == 200, r.text
        cells = _cells(zipfile.ZipFile(io.BytesIO(r.content)))
        assert 'YOLO("yolo11n.pt")' in cells[2] and "epochs=30" in cells[2]
        assert "imgsz=1280" in cells[2] and "imgsz=1280" in cells[3]


@pytest.mark.parametrize("body", [
    {"task": "detect", "base_weights": "yolo11s-seg.pt"},
    {"task": "segment", "base_weights": "yolo11s.pt"},
    {"base_weights": "yolo11s-seg.pt"},
    {"epochs": 0}, {"epochs": 501}, {"epochs": "50"}, {"epochs": None},
    {"imgsz": 1000}, {"imgsz": 4096}, {"imgsz": True},
])
def test_route_bad_options_are_400(tmp_path, body):
    with _client(tmp_path) as c:
        s = _done_project(c)
        r = c.post(f"/api/projects/{s['pid']}/train-package", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "bad_request", r.text


def test_route_bad_option_unknown_project_is_404(tmp_path):
    with _client(tmp_path) as c:
        r = c.post("/api/projects/99/train-package", json={"epochs": 0})
        assert r.status_code == 404
