"""Polygons in export (CONTRACT #14, #21): to_coco, yolo, yolo-seg, coco_for_image, the zip route."""

import copy
import json
from pathlib import Path

import pytest

from autolabel.export import build_zip, to_coco, to_yolo, to_yolo_seg
from autolabel.export.roboflow_push import coco_for_image
from tests.test_export_formats import unzip

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[1] / "docs" / "fixtures" / "api_samples.json").read_text()
)
STEM = "biax__cam0_2026-09-29_18-48-25_00290"
W, H = 1280, 720


@pytest.fixture
def data():
    """Image 11 is done; box 106 has a polygon, box 104 (bbox [300, 330, 690, 58]) has none."""
    images = copy.deepcopy(FIXTURES["GET /api/projects/1/images"]["images"])
    classes = copy.deepcopy(FIXTURES["GET /api/projects/1/classes"]["classes"])
    anns = copy.deepcopy(FIXTURES["all_annotations"])
    status = {10: "todo", 11: "done", 12: "todo"}
    for im in images:
        im["review_status"] = status[im["id"]]
    return images, classes, anns


def ann(anns, aid):
    return next(a for a in anns if a["id"] == aid)


def line_for(data, cls, seg=True):
    files = (to_yolo_seg if seg else to_yolo)(*data)
    lines = files[f"labels/{STEM}.txt"].splitlines()
    return [ln for ln in lines if ln.split()[0] == cls]


def test_coco_polygon_row_has_pixel_segmentation_and_polygon_area(data):
    images, classes, anns = data
    polygon = ann(anns, 106)["polygon"]
    a = ann(to_coco(images, classes, anns)["annotations"], 106)
    flat = [v for x, y in polygon for v in (x * W, y * H)]
    assert a["segmentation"] == [pytest.approx(flat, abs=1e-6)]
    assert a["bbox"] == [150.0, 210.0, 40.0, 50.0]
    # shoelace by hand, in pixels
    pts = [(x * W, y * H) for x, y in polygon]
    area = abs(sum(pts[i][0] * pts[(i + 1) % 5][1] - pts[(i + 1) % 5][0] * pts[i][1]
                   for i in range(5))) / 2
    assert a["area"] == pytest.approx(area, abs=1e-6)
    assert a["area"] < 40.0 * 50.0


def test_coco_box_row_unchanged(data):
    images, classes, anns = data
    a = ann(to_coco(images, classes, anns)["annotations"], 104)
    assert a["segmentation"] == []
    assert a["area"] == 690.0 * 58.0


def test_coco_polygon_wins_over_mask(data):
    images, classes, anns = data
    ann(anns, 106)["mask"] = {"size": [H, W], "counts": "abc"}
    a = ann(to_coco(images, classes, anns)["annotations"], 106)
    assert isinstance(a["segmentation"], list) and len(a["segmentation"][0]) == 10


def test_coco_row_without_polygon_key_still_works(data):
    images, classes, anns = data
    for a in anns:
        a.pop("polygon")
    assert ann(to_coco(images, classes, anns)["annotations"], 106)["segmentation"] == []


def test_yolo_polygon_row_writes_its_bbox(data):
    # class id 1 = person = index 0
    assert line_for(data, "0", seg=False) == [
        f"0 {170 / W:.6f} {235 / H:.6f} {40 / W:.6f} {50 / H:.6f}"]


def test_yolo_seg_polygon_row(data):
    assert line_for(data, "0") == ["0 0.117188 0.326389 0.132812 0.291667 0.148438 0.326389 "
                                   "0.140625 0.361111 0.125000 0.361111"]


def test_yolo_seg_box_row_is_4_corners(data):
    x0, y0, x1, y1 = 300 / W, 330 / H, 990 / W, 388 / H
    want = f"2 {x0:.6f} {y0:.6f} {x1:.6f} {y0:.6f} {x1:.6f} {y1:.6f} {x0:.6f} {y1:.6f}"
    assert line_for(data, "2") == [want]


def test_yolo_seg_box_is_clipped_or_left_out(data):
    images, classes, anns = data
    box = ann(anns, 104)
    box["bbox"] = [-10.0, 700.0, 30.0, 40.0]
    nums = [float(v) for v in line_for(data, "2")[0].split()[1:]]
    assert nums == pytest.approx([0, 700 / H, 20 / W, 700 / H, 20 / W, 1, 0, 1], abs=1e-6)
    box["bbox"] = [2000.0, 10.0, 30.0, 40.0]
    assert line_for(data, "2") == []


def test_yolo_seg_zip_has_yolo_layout(data):
    images, classes, anns = data
    files = unzip(build_zip("yolo-seg", images, classes, anns))
    assert set(files) == {"data.yaml", "classes.txt", f"labels/{STEM}.txt"}
    assert b"0 0.117188 0.326389" in files[f"labels/{STEM}.txt"]
    with_images = unzip(build_zip("yolo-seg", images, classes, anns,
                                  image_bytes=lambda im: b"jpg"))
    assert with_images[f"images/{STEM}.jpg"] == b"jpg"


IMAGE = {"id": 7, "file_name": "a.jpg", "width": 100, "height": 50}


def test_coco_for_image_mixed_rows():
    coco = coco_for_image(IMAGE, [
        {"class_name": "rb", "bbox": [10, 10, 20, 20],
         "polygon": [[0.1, 0.2], [0.3, 0.2], [0.2, 0.6]]},
        {"class_name": "person", "bbox": [40, 5, 20, 10], "polygon": None},
        {"class_name": "person", "bbox": [40, 5, 20, 10]},
    ], ["person", "rb"])
    poly, box, no_key = coco["annotations"]
    assert poly["segmentation"] == [pytest.approx([10, 10, 30, 10, 20, 30], abs=1e-6)]
    assert poly["area"] == pytest.approx(200.0, abs=1e-6)
    assert poly["bbox"] == [10.0, 10.0, 20.0, 20.0]
    assert box["segmentation"] == [] and box["area"] == 200.0
    assert no_key["segmentation"] == []
