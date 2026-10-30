"""Tests for autolabel/coco.py (merging Roboflow splits and sampling a gold set)."""

from autolabel.coco import box_counts, merge_coco, sample_coco


def split(files, cats, anns):
    """files: [name], cats: {id: name}, anns: [(image_index, cat_id)]"""
    return {
        "images": [{"id": i, "file_name": f} for i, f in enumerate(files)],
        "categories": [{"id": k, "name": v} for k, v in cats.items()],
        "annotations": [
            {"id": j, "image_id": i, "category_id": c, "bbox": [0, 0, 1, 1]} for j, (i, c) in enumerate(anns)
        ],
    }


# ---------- merge ----------

def test_merge_matches_categories_by_name_not_id():
    train = split(["a.jpg"], {0: "person", 1: "rb"}, [(0, 1)])
    valid = split(["b.jpg"], {0: "rb", 1: "person"}, [(0, 0)])  # ids swapped
    m = merge_coco([train, valid])
    assert box_counts(m) == {"rb": 2}


def test_merge_gives_unique_ids():
    m = merge_coco([split(["a.jpg"], {0: "x"}, [(0, 0)]), split(["b.jpg"], {0: "x"}, [(0, 0)])])
    assert len({im["id"] for im in m["images"]}) == 2
    assert len({a["id"] for a in m["annotations"]}) == 2


def test_merge_keeps_annotations_on_the_right_image():
    m = merge_coco([split(["a.jpg"], {0: "x"}, []), split(["b.jpg"], {0: "x"}, [(0, 0)])])
    file_of = {im["id"]: im["file_name"] for im in m["images"]}
    assert [file_of[a["image_id"]] for a in m["annotations"]] == ["b.jpg"]


def test_merge_skips_duplicate_file_names():
    m = merge_coco([split(["a.jpg"], {0: "x"}, [(0, 0)]), split(["a.jpg"], {0: "x"}, [(0, 0)])])
    assert len(m["images"]) == 1
    assert len(m["annotations"]) == 1


# ---------- sample ----------

def big_set():
    # 100 images: image 0..2 contain the rare class "rb", all contain "person".
    anns = [(i, 0) for i in range(100)] + [(i, 1) for i in range(3)]
    return split([f"{i}.jpg" for i in range(100)], {0: "person", 1: "rb"}, anns)


def test_sample_size():
    assert len(sample_coco(big_set(), 20)["images"]) == 20


def test_sample_includes_rare_class():
    s = sample_coco(big_set(), 20, target_per_class=50)
    assert box_counts(s)["rb"] == 3


def test_sample_shares_quota_between_classes():
    # The real bug: 4 classes, each on its own 60 images, n=50.
    # Before the fix the rarest class took all 50 slots.
    names = {0: "person", 1: "rb", 2: "sb", 3: "biax"}
    anns = [(i, i // 60) for i in range(240)]
    s = sample_coco(split([f"{i}.jpg" for i in range(240)], names, anns), 50, target_per_class=50)
    counts = box_counts(s)
    assert len(s["images"]) == 50
    assert all(counts[c] >= 12 for c in names.values()), counts


def test_sample_is_repeatable_with_same_seed():
    a = sample_coco(big_set(), 10, seed=1)
    b = sample_coco(big_set(), 10, seed=1)
    assert [im["file_name"] for im in a["images"]] == [im["file_name"] for im in b["images"]]


def test_sample_keeps_only_chosen_annotations():
    s = sample_coco(big_set(), 5)
    ids = {im["id"] for im in s["images"]}
    assert all(a["image_id"] in ids for a in s["annotations"])


def test_sample_bigger_than_dataset_returns_everything():
    assert len(sample_coco(big_set(), 500)["images"]) == 100


# ---------- coco_to_yolo ----------

from autolabel.coco import coco_to_yolo  # noqa: E402


def yolo_coco(anns):
    return {
        "images": [{"id": 0, "file_name": "a.jpg", "width": 200, "height": 100},
                   {"id": 1, "file_name": "empty.jpg", "width": 200, "height": 100}],
        "categories": [{"id": 7, "name": "rb"}, {"id": 9, "name": "person"}, {"id": 3, "name": "junk"}],
        "annotations": anns,
    }


def test_coco_to_yolo_normalises_and_maps_class_order():
    out = coco_to_yolo(yolo_coco([{"id": 0, "image_id": 0, "category_id": 7, "bbox": [50, 25, 100, 50]}]),
                       ["person", "rb"])
    assert out["a.jpg"] == ["1 0.500000 0.500000 0.500000 0.500000"]


def test_coco_to_yolo_keeps_empty_images():
    assert coco_to_yolo(yolo_coco([]), ["person"])["empty.jpg"] == []


def test_coco_to_yolo_skips_unknown_class():
    out = coco_to_yolo(yolo_coco([{"id": 0, "image_id": 0, "category_id": 3, "bbox": [0, 0, 10, 10]}]), ["person"])
    assert out["a.jpg"] == []


def test_coco_to_yolo_clips_to_image():
    out = coco_to_yolo(yolo_coco([{"id": 0, "image_id": 0, "category_id": 9, "bbox": [-20, 50, 60, 80]}]), ["person"])
    # clipped to x 0..40, y 50..100
    assert out["a.jpg"] == ["0 0.100000 0.750000 0.200000 0.500000"]


def test_coco_to_yolo_drops_box_outside_image():
    out = coco_to_yolo(yolo_coco([{"id": 0, "image_id": 0, "category_id": 9, "bbox": [300, 10, 20, 20]}]), ["person"])
    assert out["a.jpg"] == []
