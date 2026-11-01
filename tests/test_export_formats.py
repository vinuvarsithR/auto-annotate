"""Tests for autolabel/export (CONTRACT #14): selection rules, COCO, YOLO, zip."""

import copy
import io
import json
import zipfile
from pathlib import Path

import pytest

from autolabel.export import (
    APPROVED_STATUSES,
    NothingToExport,
    build_zip,
    to_coco,
    to_yolo,
    validate_statuses,
)

FIXTURES = json.loads(
    (Path(__file__).resolve().parents[1] / "docs" / "fixtures" / "api_samples.json").read_text()
)


@pytest.fixture
def data():
    """Fixture data with image 10 todo, 11 done, 12 empty (12 still has model boxes)."""
    images = copy.deepcopy(FIXTURES["GET /api/projects/1/images"]["images"])
    classes = copy.deepcopy(FIXTURES["GET /api/projects/1/classes"]["classes"])
    anns = copy.deepcopy(FIXTURES["all_annotations"])
    status = {10: "todo", 11: "done", 12: "empty"}
    for im in images:
        im["review_status"] = status[im["id"]]
    return images, classes, anns


def unzip(blob: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        return {n: zf.read(n) for n in zf.namelist()}


# ---- statuses ----

def test_validate_statuses_default_and_rules():
    assert validate_statuses(None) == list(APPROVED_STATUSES)
    assert validate_statuses(["accepted", "accepted", "needs_review"]) == ["accepted", "needs_review"]
    for bad in (["rejected"], ["accepted", "rejected"], ["nope"], [], "accepted"):
        with pytest.raises(ValueError):
            validate_statuses(bad)


# ---- COCO ----

def test_coco_only_done_empty_and_approved(data):
    coco = to_coco(*data)
    assert [im["id"] for im in coco["images"]] == [11, 12]
    assert [c["name"] for c in coco["categories"]] == ["person", "rb", "sb", "biax"]
    assert [c["id"] for c in coco["categories"]] == [1, 2, 3, 4]
    # image 11: 104 accepted sb, 105 edited biax, 106 human accepted person
    # image 12 is empty -> its auto_accepted box 107 is NOT exported
    assert sorted(a["id"] for a in coco["annotations"]) == [104, 105, 106]
    a105 = next(a for a in coco["annotations"] if a["id"] == 105)
    assert a105["category_id"] == 4 and a105["bbox"] == [812.0, 300.5, 240.0, 66.0]
    assert a105["area"] == pytest.approx(240.0 * 66.0)


def test_coco_status_filter(data):
    images, classes, anns = data
    images[0]["review_status"] = "done"   # image 10: 101 auto_accepted, 102 needs_review, 103 rejected
    coco = to_coco(images, classes, anns, ["needs_review"])
    assert [a["id"] for a in coco["annotations"]] == [102]
    coco = to_coco(images, classes, anns)
    assert sorted(a["id"] for a in coco["annotations"]) == [101, 104, 105, 106]


def test_rejected_never_exported(data):
    images, classes, anns = data
    for im in images:
        im["review_status"] = "done"
    for a in anns:
        a["status"] = "rejected"
    assert to_coco(images, classes, anns)["annotations"] == []
    labels = {k: v for k, v in to_yolo(images, classes, anns).items() if k.startswith("labels/")}
    assert len(labels) == 3 and all(v == "" for v in labels.values())
    with pytest.raises(ValueError):
        to_coco(images, classes, anns, ["rejected"])


def test_coco_mask_goes_to_segmentation(data):
    images, classes, anns = data
    images[2]["review_status"] = "done"   # image 12 has box 107 with a mask
    coco = to_coco(images, classes, anns)
    a107 = next(a for a in coco["annotations"] if a["id"] == 107)
    assert a107["segmentation"]["size"] == [720, 1280]
    a104 = next(a for a in coco["annotations"] if a["id"] == 104)
    assert a104["segmentation"] == []


# ---- YOLO ----

def test_yolo_files(data):
    files = to_yolo(*data)
    assert set(files) == {
        "data.yaml", "classes.txt",
        "labels/biax__cam0_2026-09-29_18-48-25_00290.txt",
        "labels/sb_evener__cam0_2026-09-29_18-57-19_00003.txt",
    }
    assert files["classes.txt"] == "person\nrb\nsb\nbiax\n"
    assert "nc: 4" in files["data.yaml"] and '["person", "rb", "sb", "biax"]' in files["data.yaml"]
    # the empty image gets an empty label file
    assert files["labels/sb_evener__cam0_2026-09-29_18-57-19_00003.txt"] == ""
    lines = files["labels/biax__cam0_2026-09-29_18-48-25_00290.txt"].splitlines()
    assert len(lines) == 3
    # box 104: sb (index 2), bbox [300, 330, 690, 58] on 1280x720
    cls, cx, cy, w, h = lines[0].split()
    assert cls == "2"
    assert float(cx) == pytest.approx((300 + 345) / 1280, abs=1e-6)
    assert float(cy) == pytest.approx((330 + 29) / 720, abs=1e-6)
    assert float(w) == pytest.approx(690 / 1280, abs=1e-6)
    assert float(h) == pytest.approx(58 / 720, abs=1e-6)


def test_yolo_clips_boxes_to_image(data):
    images, classes, anns = data
    anns = [a for a in anns if a["id"] == 104]
    anns[0]["bbox"] = [-10.0, 700.0, 30.0, 40.0]     # sticks out left and bottom
    files = to_yolo(images, classes, anns)
    cls, cx, cy, w, h = files["labels/biax__cam0_2026-09-29_18-48-25_00290.txt"].split()
    assert float(w) == pytest.approx(20 / 1280, abs=1e-6)
    assert float(h) == pytest.approx(20 / 720, abs=1e-6)
    anns[0]["bbox"] = [2000.0, 10.0, 30.0, 40.0]     # fully outside -> dropped
    assert to_yolo(images, classes, anns)["labels/biax__cam0_2026-09-29_18-48-25_00290.txt"] == ""


def test_yolo_same_stem_does_not_collide(data):
    images, classes, anns = data
    extra = dict(images[1], id=99, file_name="biax__cam0_2026-09-29_18-48-25_00290.png")
    files = to_yolo(images + [extra], classes, anns)
    assert "labels/biax__cam0_2026-09-29_18-48-25_00290.txt" in files
    assert "labels/99_biax__cam0_2026-09-29_18-48-25_00290.txt" in files


# ---- zip ----

def test_zip_coco_without_images(data):
    files = unzip(build_zip("coco", *data))
    assert set(files) == {"annotations.json"}
    coco = json.loads(files["annotations.json"])
    assert [im["id"] for im in coco["images"]] == [11, 12]


def test_zip_yolo_with_images(data):
    files = unzip(build_zip("yolo", *data, image_bytes=lambda im: f"img{im['id']}".encode()))
    assert files["images/biax__cam0_2026-09-29_18-48-25_00290.jpg"] == b"img11"
    assert files["images/sb_evener__cam0_2026-09-29_18-57-19_00003.jpg"] == b"img12"
    assert "images/rb_load__cam0_2026-09-29_18-34-11_00049.jpg" not in files   # todo image
    assert "data.yaml" in files and "labels/biax__cam0_2026-09-29_18-48-25_00290.txt" in files


def test_zip_coco_with_images(data):
    files = unzip(build_zip("coco", *data, image_bytes=lambda im: b"x"))
    assert set(files) == {
        "annotations.json",
        "images/biax__cam0_2026-09-29_18-48-25_00290.jpg",
        "images/sb_evener__cam0_2026-09-29_18-57-19_00003.jpg",
    }


def test_zip_errors(data):
    images, classes, anns = data
    with pytest.raises(ValueError):
        build_zip("voc", images, classes, anns)
    with pytest.raises(ValueError):
        build_zip("coco", images, classes, anns, ["rejected"])
    for im in images:
        im["review_status"] = "todo"
    with pytest.raises(NothingToExport):
        build_zip("coco", images, classes, anns)


def test_only_empty_images_still_export(data):
    images, classes, anns = data
    images[1]["review_status"] = "todo"     # only image 12 (empty) left
    coco = json.loads(unzip(build_zip("coco", images, classes, anns))["annotations.json"])
    assert [im["id"] for im in coco["images"]] == [12] and coco["annotations"] == []
