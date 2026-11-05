"""Polygons in plan_import (CONTRACT #22). The two fixtures come from docs/fixtures/polygons/."""

import json
from pathlib import Path

import pytest
from PIL import Image

from autolabel.importers import plan_import
from autolabel.importers.base import clamp_polygon, polygon_area, polygon_bbox

FOLDER = Path(__file__).resolve().parents[1] / "docs" / "fixtures" / "polygons"
EXPECTED = json.loads((FOLDER / "expected.json").read_text())
TOL = 1e-6


def approx_box(box):
    return {
        "class_name": box["class_name"],
        "bbox": pytest.approx(box["bbox"], abs=TOL),
        "polygon": None if box["polygon"] is None
        else [pytest.approx(p, abs=TOL) for p in box["polygon"]],
    }


@pytest.mark.parametrize("name", ["yolo_mixed", "coco_segmentation"])
def test_fixture_matches_expected(name):
    plan = plan_import(FOLDER / name)
    want = EXPECTED[name]
    assert plan.format == want["format"]
    assert plan.class_names == want["class_names"]
    assert plan.skipped == want["skipped"]
    assert [i.rel_path for i in plan.items] == [i["rel_path"] for i in want["items"]]
    for got, item in zip(plan.items, want["items"]):
        assert got.boxes == [approx_box(b) for b in item["boxes"]]


def folder(tmp_path, files, size=(100, 50)):
    for rel, text in files.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return tmp_path


def image(path, size=(100, 50)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size).save(path)


def yolo_plan(tmp_path, label):
    folder(tmp_path, {"data.yaml": "names: [a, b]", "labels/x.txt": label})
    image(tmp_path / "images" / "x.png")
    return plan_import(tmp_path)


def coco_plan(tmp_path, *annotations):
    image(tmp_path / "x.png")
    folder(tmp_path, {"c.json": json.dumps({
        "images": [{"id": 1, "file_name": "x.png"}],
        "categories": [{"id": 1, "name": "a"}],
        "annotations": [{"id": i, "image_id": 1, "category_id": 1, **a}
                        for i, a in enumerate(annotations, start=1)],
    })})
    return plan_import(tmp_path)


def test_yolo_polygon_with_4_points_is_not_read_as_a_box(tmp_path):
    plan = yolo_plan(tmp_path, "0 0.1 0.2 0.3 0.2 0.3 0.6 0.1 0.6\n")
    box = plan.items[0].boxes[0]
    assert box["polygon"] == [[0.1, 0.2], [0.3, 0.2], [0.3, 0.6], [0.1, 0.6]]
    assert box["bbox"] == pytest.approx([10, 10, 20, 20], abs=TOL)


def test_yolo_box_rows_have_polygon_none(tmp_path):
    plan = yolo_plan(tmp_path, "0 0.5 0.5 0.2 0.4\n")
    assert plan.items[0].boxes[0]["polygon"] is None


def test_yolo_collinear_polygon_with_float_noise_has_no_area(tmp_path):
    plan = yolo_plan(tmp_path, "0 0.1 0.1 0.2 0.2 0.3 0.3\n")
    assert plan.items[0].boxes == []
    assert plan.skipped == [{"file_name": "labels/x.txt", "reason": "line 1: polygon has no area"}]


def test_yolo_unknown_class_on_polygon_line(tmp_path):
    plan = yolo_plan(tmp_path, "7 0.1 0.2 0.3 0.2 0.2 0.6\n")
    assert plan.skipped == [{"file_name": "labels/x.txt", "reason": "line 1: unknown class index 7"}]


def test_coco_no_note_for_plain_boxes(tmp_path):
    plan = coco_plan(tmp_path, {"bbox": [1, 2, 3, 4]}, {"bbox": [1, 2, 3, 4], "segmentation": []})
    assert plan.skipped == []
    assert [b["polygon"] for b in plan.items[0].boxes] == [None, None]


def test_coco_one_note_per_file(tmp_path):
    image(tmp_path / "train" / "x.png")
    image(tmp_path / "valid" / "y.png")
    for split, name in (("train", "x.png"), ("valid", "y.png")):
        folder(tmp_path, {f"{split}/_annotations.coco.json": json.dumps({
            "images": [{"id": 1, "file_name": name}],
            "categories": [{"id": 1, "name": "a"}],
            "annotations": [{"id": i, "image_id": 1, "category_id": 1, "bbox": [1, 2, 3, 4],
                             "segmentation": {"size": [50, 100], "counts": "x"}}
                            for i in (1, 2)],
        })})
    plan = plan_import(tmp_path)
    note = "2 segmentations imported as box only (RLE, multi-part or unusable polygon)"
    assert plan.skipped == [{"file_name": "train/_annotations.coco.json", "reason": note},
                            {"file_name": "valid/_annotations.coco.json", "reason": note}]


def test_coco_row_not_stored_is_not_counted(tmp_path):
    plan = coco_plan(tmp_path,
                     {"bbox": [500, 500, 3, 4], "segmentation": {"counts": "x"}},  # outside image
                     {"bbox": [0, 0, "x", 1], "segmentation": [[1, 2]]})           # bad bbox
    assert [s["reason"] for s in plan.skipped] == ["annotation 2: bad bbox"]


def test_coco_polygon_outside_image_falls_back_to_box(tmp_path):
    plan = coco_plan(tmp_path, {"bbox": [1, 2, 3, 4],
                                "segmentation": [[200, 10, 300, 10, 250, 40]]})
    assert plan.items[0].boxes[0]["polygon"] is None
    assert plan.items[0].boxes[0]["bbox"] == [1.0, 2.0, 3.0, 4.0]
    assert "1 segmentations imported as box only" in plan.skipped[0]["reason"]


def test_coco_polygon_replaces_a_wrong_bbox(tmp_path):
    plan = coco_plan(tmp_path, {"bbox": [0, 0, 99, 49], "segmentation": [[10, 10, 30, 10, 20, 30]]})
    assert plan.items[0].boxes[0]["bbox"] == pytest.approx([10, 10, 20, 20], abs=TOL)


def test_polygon_helpers():
    square = [[0, 0], [2, 0], [2, 2], [0, 2]]
    assert polygon_area(square) == 4
    assert polygon_area(square[::-1]) == 4
    assert clamp_polygon([[-1, 0.5], [2, 0.5], [0.5, 3]]) == [[0, 0.5], [1, 0.5], [0.5, 1]]
    assert clamp_polygon([[2, 0], [3, 0.5], [4, 1]]) is None
    assert polygon_bbox([[0.1, 0.2], [0.3, 0.2], [0.2, 0.6]], 100, 50) == pytest.approx(
        [10, 10, 20, 20], abs=TOL)
