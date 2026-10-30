"""Tests for autolabel/split.py: time windows, test picking and leak checks, on synthetic videos."""

import random

from autolabel.split import (
    GAP, TEST, TRAIN, VAL, Bounds, Windows, assign_windows, build_split, check_no_leak,
    make_test_accept, window_bounds, window_of,
)
from autolabel.sequences import parse_frame_name

CLASSES = ["person", "rb", "sb", "biax"]


def name(seq, frame):
    return f"{seq}__cam0_2026-09-29_10-00-00_{frame:05d}_jpg.rf.abc{frame}.jpg"


# ---------- windows ----------

def test_window_bounds_on_100_frames():
    b = window_bounds(0, 99, Windows())
    assert b == Bounds(val_end=10, train_start=25, train_end=65, test_start=80)  # gap = max(5, 15) = 15


def test_long_video_uses_percentage_gap():
    b = window_bounds(0, 999, Windows())
    assert b.train_start - b.val_end == 50
    assert b.test_start - b.train_end == 50


def test_window_of_each_region():
    b = window_bounds(0, 99)
    assert window_of(0, b) == VAL
    assert window_of(10, b) == GAP
    assert window_of(25, b) == TRAIN
    assert window_of(64, b) == TRAIN
    assert window_of(65, b) == GAP
    assert window_of(80, b) == TEST
    assert window_of(None, b) == GAP


def test_windows_use_each_videos_own_range():
    names = [name("a", f) for f in range(0, 100)] + [name("b", f) for f in range(500, 600)]
    w = assign_windows(names)
    assert w[name("a", 85)] == TEST
    assert w[name("b", 585)] == TEST
    assert w[name("b", 30 + 500)] == TRAIN


def test_frame_without_number_is_left_out():
    assert assign_windows(["snapshot.jpg"]) == {"snapshot.jpg": GAP}


# ---------- test-image veto ----------

def test_accept_enforces_cap_gap_and_lookalikes():
    hashes = {name("a", f): f * 1000003 for f in range(200)}  # far-apart hashes
    hashes[name("a", 150)] = hashes[name("a", 100)]  # a look-alike
    accept = make_test_accept(hashes, max_per_seq=2, min_test_gap=10, max_dist=4)
    chosen = [{"file_name": name("a", 100)}]
    assert not accept({"file_name": name("a", 105)}, chosen)  # too close in time
    assert not accept({"file_name": name("a", 150)}, chosen)  # looks the same
    assert accept({"file_name": name("a", 120)}, chosen)
    chosen.append({"file_name": name("a", 120)})
    assert not accept({"file_name": name("a", 190)}, chosen)  # cap of 2 reached


# ---------- leak check ----------

def test_check_no_leak_passes_clean_split():
    split = {name("a", 5): VAL, name("a", 50): TRAIN, name("a", 90): TEST}
    assert check_no_leak(split, min_gap_frames=15) == []


def test_check_no_leak_flags_close_frames():
    split = {name("a", 80): TRAIN, name("a", 90): TEST}
    problems = check_no_leak(split, min_gap_frames=15)
    assert len(problems) == 1 and "train" in problems[0]


def test_check_no_leak_flags_lookalikes_across_videos():
    split = {name("a", 10): TRAIN, name("b", 90): TEST}
    hashes = {name("a", 10): 0b1111, name("b", 90): 0b1110}
    assert check_no_leak(split, 15, hashes, max_dist=2)
    assert check_no_leak(split, 15) == []  # without hashes: different videos are fine


# ---------- whole pipeline on synthetic data ----------

def synthetic_coco(seed=0):
    """3 videos x 200 frames; sb only in video 's', biax only in 'b' (like our data)."""
    rnd = random.Random(seed)
    images, anns = [], []
    cats = [{"id": i, "name": c} for i, c in enumerate(CLASSES)]
    for seq, extra in (("r", "rb"), ("s", "sb"), ("b", "biax")):
        for f in range(200):
            iid = len(images)
            images.append({"id": iid, "file_name": name(seq, f), "width": 100, "height": 100})
            for cls in ("person", extra):
                anns.append({"id": len(anns), "image_id": iid, "category_id": CLASSES.index(cls), "bbox": [1, 1, 5, 5]})
    hashes = {im["file_name"]: rnd.getrandbits(64) for im in images}
    return {"images": images, "categories": cats, "annotations": anns}, hashes


def split_map(result):
    return {im["file_name"]: s for s in (TRAIN, VAL, TEST) for im in result[s]["images"]}


def test_build_split_has_no_leaks_and_covers_all_classes():
    coco, hashes = synthetic_coco()
    r = build_split(coco, hashes, n_test=12, max_per_seq=4, min_test_gap=5, max_dist=3)
    sm = split_map(r)
    assert check_no_leak(sm, Windows().min_gap_frames, hashes, max_dist=3) == []

    cat = {c["id"]: c["name"] for c in coco["categories"]}
    test_ids = {im["id"] for im in r[TEST]["images"]}
    test_classes = {cat[a["category_id"]] for a in r[TEST]["annotations"] if a["image_id"] in test_ids}
    assert test_classes == set(CLASSES)


def test_no_test_window_frame_is_used_for_training():
    coco, hashes = synthetic_coco()
    r = build_split(coco, hashes, n_test=6, max_per_seq=2, min_test_gap=5, max_dist=3)
    for s in (TRAIN, VAL):
        for im in r[s]["images"]:
            assert r["windows"][im["file_name"]] == s


def test_test_images_respect_cap_and_gap():
    coco, hashes = synthetic_coco()
    r = build_split(coco, hashes, n_test=12, max_per_seq=4, min_test_gap=5, max_dist=3)
    frames = {}
    for im in r[TEST]["images"]:
        seq, f = parse_frame_name(im["file_name"])
        frames.setdefault(seq, []).append(f)
    for fs in frames.values():
        assert len(fs) <= 4
        fs.sort()
        assert all(b - a >= 5 for a, b in zip(fs, fs[1:]))


def test_train_lookalike_of_test_is_dropped():
    coco, hashes = synthetic_coco()
    train_name = name("r", 100)
    r0 = build_split(coco, hashes, n_test=3, max_per_seq=1, min_test_gap=5, max_dist=3)
    t = r0[TEST]["images"][0]["file_name"]
    hashes[train_name] = hashes[t]  # make a train frame identical to a test frame
    r = build_split(coco, hashes, n_test=3, max_per_seq=1, min_test_gap=5, max_dist=3)
    assert train_name in r["dropped_lookalike"]
    assert train_name not in split_map(r)
