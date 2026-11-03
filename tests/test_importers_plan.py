"""plan_import: format detection + COCO / YOLO / images parsing (CONTRACT #22).

Small datasets are built in tmp_path. A zip and a folder upload (list of relative path + bytes
pairs) must give the same plan.
"""

import io
import json
import zipfile

import pytest
from PIL import Image

from autolabel.importers import ImportItem, check_names, check_zip, extract_zip, plan_import


def img(w=100, h=50, fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, fmt)
    return buf.getvalue()


def tree(root, files):
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data if isinstance(data, bytes) else data.encode())
    return root


def coco(images, annotations, categories):
    return json.dumps({"info": {}, "images": images, "annotations": annotations,
                       "categories": categories})


def items(plan):
    return {it.rel_path: it.boxes for it in plan.items}


def reasons(plan):
    return [(s["file_name"], s["reason"]) for s in plan.skipped]


# ---------- images ----------

def test_plain_images(tmp_path):
    root = tree(tmp_path, {"b.png": img(), "a/c.JPG": img(fmt="JPEG"), "d.jpeg": img(fmt="JPEG"),
                           "notes.txt": "hi", "x.gif": img(fmt="GIF")})
    plan = plan_import(root)
    assert plan.format == "images"
    assert plan.items == [ImportItem("a/c.JPG", []), ImportItem("b.png", []), ImportItem("d.jpeg", [])]
    assert plan.class_names == []
    assert reasons(plan) == [("notes.txt", "not an image"), ("x.gif", "not an image")]


def test_broken_image_is_skipped(tmp_path):
    root = tree(tmp_path, {"good.png": img(), "bad.jpg": b"not really a jpeg"})
    plan = plan_import(root)
    assert [i.rel_path for i in plan.items] == ["good.png"]
    assert reasons(plan) == [("bad.jpg", "cannot open image")]


def test_empty_folder(tmp_path):
    plan = plan_import(tmp_path)
    assert (plan.format, plan.items, plan.class_names, plan.skipped) == ("images", [], [], [])


# ---------- COCO ----------

CATS = [{"id": 0, "name": "crane-objects", "supercategory": "none"},
        {"id": 1, "name": "person", "supercategory": "crane-objects"},
        {"id": 2, "name": "biax", "supercategory": "crane-objects"},
        {"id": 3, "name": "rb", "supercategory": "crane-objects"}]


def test_coco_single_file(tmp_path):
    root = tree(tmp_path, {
        "images/a.png": img(100, 50),
        "images/b.png": img(100, 50),
        "annotations.json": coco(
            [{"id": 1, "file_name": "images/a.png", "width": 100, "height": 50},
             {"id": 2, "file_name": "b.png", "width": 100, "height": 50}],
            [{"id": 10, "image_id": 1, "category_id": 2, "bbox": [10, 5, 20, 10], "score": 0.3,
              "segmentation": [[1, 2, 3, 4]]},
             {"id": 11, "image_id": 1, "category_id": 1, "bbox": [90, 40, 30, 30]},  # clipped
             {"id": 12, "image_id": 2, "category_id": 1, "bbox": [200, 200, 5, 5]}],  # outside
            CATS),
    })
    plan = plan_import(root)
    assert plan.format == "coco"
    assert items(plan) == {
        "images/a.png": [{"class_name": "biax", "bbox": [10.0, 5.0, 20.0, 10.0]},
                         {"class_name": "person", "bbox": [90.0, 40.0, 10.0, 10.0]}],
        "images/b.png": [],
    }
    # file order (categories order), only names with a box; super-category and rb not created
    assert plan.class_names == ["person", "biax"]
    assert plan.skipped == []


def test_coco_one_json_per_roboflow_split(tmp_path):
    cats = CATS
    root = tree(tmp_path, {
        "README.roboflow.txt": "exported",
        "train/f_1.jpg": img(fmt="JPEG"),
        "train/same.jpg": img(fmt="JPEG"),
        "train/_annotations.coco.json": coco(
            [{"id": 0, "file_name": "f_1.jpg"}, {"id": 1, "file_name": "same.jpg"}],
            [{"id": 0, "image_id": 0, "category_id": 3, "bbox": [1, 1, 2, 2]},
             {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4]}], cats),
        "valid/same.jpg": img(fmt="JPEG"),
        "valid/_annotations.coco.json": coco(
            [{"id": 0, "file_name": "same.jpg"}],
            [{"id": 0, "image_id": 0, "category_id": 2, "bbox": [5, 5, 5, 5]}], cats),
    })
    plan = plan_import(root)
    assert plan.format == "coco"
    # same base name in two splits: each split's json applies to its own folder only
    assert items(plan) == {
        "train/f_1.jpg": [{"class_name": "rb", "bbox": [1.0, 1.0, 2.0, 2.0]}],
        "train/same.jpg": [{"class_name": "person", "bbox": [0.0, 0.0, 4.0, 4.0]}],
        "valid/same.jpg": [{"class_name": "biax", "bbox": [5.0, 5.0, 5.0, 5.0]}],
    }
    assert plan.class_names == ["person", "biax", "rb"]  # categories order of the files
    assert reasons(plan) == [("README.roboflow.txt", "not an image")]


def test_coco_nested_json_wins_over_root(tmp_path):
    cats = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]
    root = tree(tmp_path, {
        "all.json": coco([{"id": 1, "file_name": "x.png"}, {"id": 2, "file_name": "y.png"}],
                         [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]},
                          {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1]}], cats),
        "sub/x.png": img(),
        "sub/sub.json": coco([{"id": 7, "file_name": "x.png"}],
                             [{"id": 1, "image_id": 7, "category_id": 2, "bbox": [0, 0, 2, 2]}], cats),
        "y.png": img(),
    })
    plan = plan_import(root)
    assert items(plan) == {"sub/x.png": [{"class_name": "b", "bbox": [0.0, 0.0, 2.0, 2.0]}],
                           "y.png": [{"class_name": "a", "bbox": [0.0, 0.0, 1.0, 1.0]}]}
    # all.json's x.png entry had no image left for it
    assert reasons(plan) == [("all.json", "no image file for 'x.png'")]


def test_coco_bad_annotations_and_missing_images(tmp_path):
    root = tree(tmp_path, {
        "a.png": img(),
        "broken.png": b"nope",
        "c.json": coco(
            [{"id": 1, "file_name": "a.png"}, {"id": 2, "file_name": "gone.png"},
             {"id": 3, "file_name": "broken.png"}],
            [{"id": 1, "image_id": 1, "category_id": 9, "bbox": [0, 0, 1, 1]},
             {"id": 2, "image_id": 1, "category_id": 1, "bbox": [0, 0, "x", 1]},
             {"id": 3, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1]},
             {"id": 4, "image_id": 2, "category_id": 1, "bbox": [0, 0, 1, 1]},   # missing image
             {"id": 5, "image_id": 3, "category_id": 1, "bbox": [0, 0, 1, 1]},   # broken image
             {"id": 6, "image_id": 99, "category_id": 1, "bbox": [0, 0, 1, 1]},
             {"id": 7, "image_id": 1, "category_id": 1, "bbox": [3, 3, 0, 5]},   # no area
             {"id": 8, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4]}],
            [{"id": 1, "name": "person"}, {"id": 2, "name": "unused"}]),
        "other.json": json.dumps({"not": "coco"}),
        "bad.json": "{not json",
    })
    plan = plan_import(root)
    assert plan.format == "coco"
    assert items(plan) == {"a.png": [{"class_name": "person", "bbox": [1.0, 2.0, 3.0, 4.0]}]}
    assert plan.class_names == ["person"]
    assert reasons(plan) == [
        ("bad.json", "not an image"),
        ("broken.png", "cannot open image"),
        ("c.json", "no image file for 'gone.png'"),
        ("c.json", "annotation 1: unknown category_id 9"),
        ("c.json", "annotation 2: bad bbox"),
        ("c.json", "annotation 3: bad bbox"),
        ("c.json", "annotation 6: unknown image_id 99"),
        ("other.json", "not an image"),
    ]


def test_coco_wins_over_yolo(tmp_path):
    root = tree(tmp_path, {
        "images/a.png": img(), "labels/a.txt": "0 0.5 0.5 0.2 0.2", "data.yaml": "names: [p]",
        "c.json": coco([{"id": 1, "file_name": "a.png"}], [], [{"id": 1, "name": "p"}]),
    })
    plan = plan_import(root)
    assert plan.format == "coco"
    assert items(plan) == {"images/a.png": []}
    assert plan.class_names == []
    assert reasons(plan) == [("data.yaml", "not an image"), ("labels/a.txt", "not an image")]


# ---------- YOLO ----------

def roboflow_yolo():
    return {
        "data.yaml": "train: ../train/images\nval: ../valid/images\nnc: 3\nnames: ['person', 'biax', 'rb']\n",
        "README.roboflow.txt": "x",
        "train/images/f_1.jpg": img(200, 100, "JPEG"),
        "train/labels/f_1.txt": "1 0.5 0.5 0.1 0.2\n0 0.25 0.5 0.5 1.0\n\n",
        "train/images/f_2.png": img(200, 100),       # no label file -> no boxes
        "valid/images/f_3.jpg": img(200, 100, "JPEG"),
        "valid/labels/f_3.txt": "1 0.95 0.5 0.2 0.2\n",  # crosses the right edge -> clipped
        "test/images/f_4.jpg": img(200, 100, "JPEG"),
        "test/labels/f_4.txt": "",                   # empty label file -> no boxes
    }


def test_yolo_roboflow_split_folders(tmp_path):
    plan = plan_import(tree(tmp_path, roboflow_yolo()))
    assert plan.format == "yolo"
    got = items(plan)
    assert list(got) == ["test/images/f_4.jpg", "train/images/f_1.jpg", "train/images/f_2.png",
                         "valid/images/f_3.jpg"]
    assert got["train/images/f_1.jpg"] == [
        {"class_name": "biax", "bbox": pytest.approx([90.0, 40.0, 20.0, 20.0])},
        {"class_name": "person", "bbox": pytest.approx([0.0, 0.0, 100.0, 100.0])},
    ]
    assert got["valid/images/f_3.jpg"] == [
        {"class_name": "biax", "bbox": pytest.approx([170.0, 40.0, 30.0, 20.0])}]
    assert got["train/images/f_2.png"] == [] and got["test/images/f_4.jpg"] == []
    assert plan.class_names == ["person", "biax"]  # names order; rb has no box
    assert reasons(plan) == [("README.roboflow.txt", "not an image")]


def test_yolo_names_as_dict_and_classes_txt(tmp_path):
    root = tree(tmp_path, {"data.yaml": "names:\n  0: cat\n  1: dog\n",
                           "images/a.png": img(10, 10), "labels/a.txt": "1 0.5 0.5 0.2 0.2"})
    plan = plan_import(root)
    assert plan.format == "yolo" and plan.class_names == ["dog"]

    other = tree(tmp_path / "t", {"classes.txt": "cat\ndog\n\n",
                                  "images/a.png": img(10, 10), "labels/a.txt": "0 0.5 0.5 1 1"})
    plan = plan_import(other)
    assert plan.format == "yolo"
    assert items(plan) == {"images/a.png": [{"class_name": "cat", "bbox": [0.0, 0.0, 10.0, 10.0]}]}


def test_yolo_top_most_config_is_used(tmp_path):
    root = tree(tmp_path, {"classes.txt": "top\n", "sub/data.yaml": "names: [deep]",
                           "sub/images/a.png": img(10, 10), "sub/labels/a.txt": "0 0.5 0.5 1 1"})
    plan = plan_import(root)
    assert plan.class_names == ["top"]
    assert reasons(plan) == [("sub/data.yaml", "not an image")]


def test_yolo_unreadable_config_falls_to_next(tmp_path):
    root = tree(tmp_path, {"data.yaml": "names: [unclosed", "sub/classes.txt": "ok\n",
                           "images/a.png": img(10, 10), "labels/a.txt": "0 0.5 0.5 1 1"})
    plan = plan_import(root)
    assert plan.format == "yolo" and plan.class_names == ["ok"]
    assert reasons(plan) == [("data.yaml", "could not read class names")]


def test_yolo_bad_lines_and_orphan_labels(tmp_path):
    root = tree(tmp_path, {
        "data.yaml": "names: [a, b]",
        "images/x.png": img(10, 10),
        "labels/x.txt": ("0 0.5 0.5 0.2 0.2\n"
                         "7 0.5 0.5 0.2 0.2\n"
                         "0 0.5 0.5\n"
                         "a 0.5 0.5 0.2 0.2\n"
                         "0 nan 0.5 0.2 0.2\n"
                         "1.5 0.5 0.5 0.2 0.2\n"
                         "1 0.5 0.5 0.0 0.3\n"       # no area -> dropped quietly
                         "1 0.1 0.1 0.2 0.2 0.9\n"),
        "labels/orphan.txt": "0 0.5 0.5 1 1",
        "labels/notes.md": "x",
    })
    plan = plan_import(root)
    assert items(plan) == {"images/x.png": [{"class_name": "a", "bbox": pytest.approx([4.0, 4.0, 2.0, 2.0])}]}
    assert plan.class_names == ["a"]
    assert reasons(plan) == [
        ("labels/notes.md", "not an image"),
        ("labels/orphan.txt", "no matching image"),
        ("labels/x.txt", "line 2: unknown class index 7"),
        ("labels/x.txt", "line 3: expected 5 values, got 3"),
        ("labels/x.txt", "line 4: not a number"),
        ("labels/x.txt", "line 5: not a number"),
        ("labels/x.txt", "line 6: not a number"),
        ("labels/x.txt", "line 8: expected 5 values, got 6"),
    ]


def test_yolo_needs_class_names(tmp_path):
    root = tree(tmp_path, {"images/a.png": img(), "labels/a.txt": "0 0.5 0.5 1 1"})
    plan = plan_import(root)
    assert plan.format == "images"
    assert items(plan) == {"images/a.png": []}
    assert reasons(plan) == [("labels/a.txt", "not an image")]


def test_yolo_needs_images_folder_next_to_labels(tmp_path):
    root = tree(tmp_path, {"data.yaml": "names: [a]", "pics/a.png": img(), "labels/a.txt": "0 .5 .5 1 1"})
    plan = plan_import(root)
    assert plan.format == "images"


def test_yolo_image_outside_images_folder_has_no_boxes(tmp_path):
    root = tree(tmp_path, {"data.yaml": "names: [a]", "images/a.png": img(10, 10),
                           "labels/a.txt": "0 .5 .5 1 1", "extra/a.png": img(10, 10)})
    plan = plan_import(root)
    assert items(plan) == {"extra/a.png": [],
                           "images/a.png": [{"class_name": "a", "bbox": [0.0, 0.0, 10.0, 10.0]}]}


def test_yolo_label_of_broken_image_not_reported_twice(tmp_path):
    root = tree(tmp_path, {"data.yaml": "names: [a]", "images/a.png": b"broken",
                           "labels/a.txt": "0 .5 .5 1 1"})
    plan = plan_import(root)
    assert plan.items == [] and plan.class_names == []
    assert reasons(plan) == [("images/a.png", "cannot open image")]


# ---------- zip vs folder upload ----------

@pytest.mark.parametrize("files", [roboflow_yolo(), {
    "train/a.png": img(), "train/_annotations.coco.json": coco(
        [{"id": 1, "file_name": "a.png"}], [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 5, 5]}],
        [{"id": 1, "name": "p"}]),
    "loose.txt": "x"}])
def test_zip_and_folder_upload_give_same_plan(tmp_path, files):
    # zip: check -> extract -> plan
    z = tmp_path / "u.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for rel, data in files.items():
            zf.writestr(rel, data)
    n_images = check_zip(z, 10**9)
    assert extract_zip(z, tmp_path / "zip", 10**9) == []
    from_zip = plan_import(tmp_path / "zip")

    # folder upload: (relative path, bytes) pairs -> check -> save at relative paths -> plan
    pairs = list(files.items())
    check_names([rel for rel, _ in pairs])
    from_parts = plan_import(tree(tmp_path / "parts", dict(pairs)))

    assert from_zip == from_parts
    assert len(from_zip.items) == n_images
