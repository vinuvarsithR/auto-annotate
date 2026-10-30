"""
split.py - split video frames into train / val / test without leaking.

Plain-English summary
---------------------
Neighbouring video frames look almost the same. If one lands in train and
its neighbour in test, the test score is flattered. Our data has too few
videos to keep whole videos apart (sb is in one video only), so each video
is cut BY TIME instead:

    [ val 10% ][ gap ][ -------- train -------- ][ gap ][ test 20% ]

* Gaps are thrown away: no train/val frame sits right next to a test frame.
  A gap is 5% of the video's frame range, but at least `min_gap_frames`.
* Test images are picked from the test window only, with the class
  round-robin from sample_coco, at most `max_per_seq` per video, at least
  `min_test_gap` frames apart, and no two that look alike (dHash).
* Unpicked test-window frames are NOT reused for training.
* Any train/val image that looks like a test image (dHash) is dropped.

check_no_leak() re-checks the finished split from scratch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from autolabel.coco import sample_coco
from autolabel.dedup import hamming
from autolabel.sequences import parse_frame_name

VAL, TRAIN, TEST, GAP = "val", "train", "test", "gap"


@dataclass(frozen=True)
class Windows:
    val: float = 0.10
    gap: float = 0.05
    test: float = 0.20
    min_gap_frames: int = 15


@dataclass(frozen=True)
class Bounds:
    """Frame cut-offs for one video. val: f < val_end; train: train_start <= f < train_end; test: f >= test_start."""

    val_end: int
    train_start: int
    train_end: int
    test_start: int


def window_bounds(first: int, last: int, w: Windows = Windows()) -> Bounds:
    span = last - first + 1
    gap = max(round(w.gap * span), w.min_gap_frames)
    val_end = first + round(w.val * span)
    test_start = last + 1 - round(w.test * span)
    return Bounds(val_end, val_end + gap, test_start - gap, test_start)


def window_of(frame: int | None, b: Bounds) -> str:
    if frame is None:
        return GAP  # no frame number: can't prove it's far from test, so leave it out
    if frame >= b.test_start:
        return TEST
    if frame < b.val_end:
        return VAL
    if b.train_start <= frame < b.train_end:
        return TRAIN
    return GAP


def assign_windows(file_names: list[str], w: Windows = Windows()) -> dict[str, str]:
    """file name -> val / train / test / gap, using each video's own first and last frame."""
    frames: dict[str, list[int]] = defaultdict(list)
    parsed = {n: parse_frame_name(n) for n in file_names}
    for seq, f in parsed.values():
        if f is not None:
            frames[seq].append(f)
    bounds = {seq: window_bounds(min(fs), max(fs), w) for seq, fs in frames.items()}
    return {n: (window_of(f, bounds[seq]) if f is not None else GAP) for n, (seq, f) in parsed.items()}


def make_test_accept(hashes: dict[str, int], max_per_seq: int, min_test_gap: int, max_dist: int):
    """Veto function for sample_coco: per-video cap, frame gap, and no look-alikes."""

    def accept(image: dict, chosen: list[dict]) -> bool:
        seq, frame = parse_frame_name(image["file_name"])
        same_seq = 0
        for c in chosen:
            cseq, cframe = parse_frame_name(c["file_name"])
            if cseq == seq:
                same_seq += 1
                if frame is None or cframe is None or abs(frame - cframe) < min_test_gap:
                    return False
            if hamming(hashes[image["file_name"]], hashes[c["file_name"]]) <= max_dist:
                return False
        return same_seq < max_per_seq

    return accept


def _subset(coco: dict, names: set[str]) -> dict:
    ids = {im["id"] for im in coco["images"] if im["file_name"] in names}
    return {
        "images": [im for im in coco["images"] if im["id"] in ids],
        "categories": coco["categories"],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
    }


def build_split(
    coco: dict,
    hashes: dict[str, int],
    n_test: int = 50,
    w: Windows = Windows(),
    max_per_seq: int = 10,
    min_test_gap: int = 10,
    max_dist: int = 24,
    seed: int = 0,
) -> dict:
    """
    Returns {"train": coco, "val": coco, "test": coco,
             "windows": {file: window}, "dropped_lookalike": [file, ...]}.
    """
    windows = assign_windows([im["file_name"] for im in coco["images"]], w)

    test_pool = _subset(coco, {n for n, win in windows.items() if win == TEST})
    accept = make_test_accept(hashes, max_per_seq, min_test_gap, max_dist)
    test = sample_coco(test_pool, n_test, target_per_class=n_test, seed=seed, accept=accept)
    test_names = [im["file_name"] for im in test["images"]]

    dropped = []
    keep: dict[str, set[str]] = {TRAIN: set(), VAL: set()}
    for name, win in windows.items():
        if win not in keep:
            continue
        if any(hamming(hashes[name], hashes[t]) <= max_dist for t in test_names):
            dropped.append(name)
        else:
            keep[win].add(name)

    return {
        TRAIN: _subset(coco, keep[TRAIN]),
        VAL: _subset(coco, keep[VAL]),
        TEST: test,
        "windows": windows,
        "dropped_lookalike": sorted(dropped),
    }


def check_no_leak(
    split_of: dict[str, str], min_gap_frames: int, hashes: dict[str, int] | None = None, max_dist: int = 24
) -> list[str]:
    """
    split_of: file name -> 'train' / 'val' / 'test' (one entry per file, so a
    file can't be in two splits). Returns a list of problems; empty means clean.
    Checks: every train/val frame is at least min_gap_frames from every test
    frame of the same video, val and train are min_gap_frames apart, and
    (if hashes are given) no train/val image looks like a test image.
    """
    by_seq: dict[str, dict[str, list[tuple[int | None, str]]]] = defaultdict(lambda: defaultdict(list))
    for name, split in split_of.items():
        seq, frame = parse_frame_name(name)
        by_seq[seq][split].append((frame, name))

    problems = []
    for seq, parts in by_seq.items():
        for a, b in ((TRAIN, TEST), (VAL, TEST), (VAL, TRAIN)):
            for fa, na in parts.get(a, []):
                for fb, nb in parts.get(b, []):
                    if fa is None or fb is None or abs(fa - fb) < min_gap_frames:
                        problems.append(f"{a} {na} is within {min_gap_frames} frames of {b} {nb}")

    if hashes is not None:
        tests = [n for n, s in split_of.items() if s == TEST]
        for name, split in split_of.items():
            if split == TEST:
                continue
            for t in tests:
                if hamming(hashes[name], hashes[t]) <= max_dist:
                    problems.append(f"{split} {name} looks like test {t}")
    return problems
