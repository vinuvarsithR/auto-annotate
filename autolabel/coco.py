"""
coco.py - small helpers for COCO annotation files.

merge_coco:   Roboflow exports one COCO file per split (train/valid/test).
              This joins them into one file with fresh, non-clashing ids.
sample_coco:  pick N images for a gold set, making sure rare classes are
              represented before filling up with random images.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable


def merge_coco(parts: list[dict]) -> dict:
    """
    Join several COCO dicts into one. Categories are matched by NAME
    (each split numbers its categories on its own). Images with a file name
    already seen are skipped, along with their annotations.
    """
    categories: dict[str, int] = {}
    images, annotations = [], []
    seen_files: set[str] = set()

    for part in parts:
        cat_id_to_name = {c["id"]: c["name"] for c in part.get("categories", [])}
        for name in cat_id_to_name.values():
            categories.setdefault(name, len(categories))

        img_id_map = {}
        for im in part.get("images", []):
            if im["file_name"] in seen_files:
                continue
            seen_files.add(im["file_name"])
            new_id = len(images)
            img_id_map[im["id"]] = new_id
            images.append({**im, "id": new_id})

        for a in part.get("annotations", []):
            if a["image_id"] not in img_id_map:
                continue
            annotations.append(
                {
                    **a,
                    "id": len(annotations),
                    "image_id": img_id_map[a["image_id"]],
                    "category_id": categories[cat_id_to_name[a["category_id"]]],
                }
            )

    return {
        "images": images,
        "categories": [{"id": i, "name": n} for n, i in categories.items()],
        "annotations": annotations,
    }


def sample_coco(
    coco: dict,
    n: int,
    target_per_class: int = 50,
    seed: int = 0,
    accept: Callable[[dict, list[dict]], bool] | None = None,
) -> dict:
    """
    Choose up to n images:
      1. Classes take turns (rarest first): each round, every class with
         fewer than target_per_class boxes gets one more image containing it,
         picked in a random but seeded (so repeatable) order.
      2. Fill the rest randomly, including images with no boxes (they test
         whether the model draws junk on empty scenes).
    accept(image, chosen_images), if given, can veto an image (e.g. too close
    to one already chosen); vetoed images are skipped, not retried.
    Returns a new COCO dict with only the chosen images and their boxes.
    """
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    boxes_per_image: dict[int, Counter] = defaultdict(Counter)
    for a in coco["annotations"]:
        boxes_per_image[a["image_id"]][cat_name[a["category_id"]]] += 1

    image_by_id = {im["id"]: im for im in coco["images"]}
    order = [im["id"] for im in coco["images"]]
    random.Random(seed).shuffle(order)

    chosen: list[int] = []
    chosen_set: set[int] = set()
    collected: Counter = Counter()

    def ok(img_id: int) -> bool:
        return accept is None or accept(image_by_id[img_id], [image_by_id[i] for i in chosen])

    # Classes take turns, rarest first: each round, every class still under its
    # target gets one more image. No single class can use up the whole quota.
    totals = box_counts(coco)
    classes = sorted(totals, key=lambda c: totals[c])
    candidates = {c: iter([i for i in order if boxes_per_image[i][c] > 0]) for c in classes}
    progress = True
    while progress and len(chosen) < n:
        progress = False
        for cls in classes:
            if len(chosen) >= n:
                break
            if collected[cls] >= target_per_class:
                continue
            for img_id in candidates[cls]:
                if img_id not in chosen_set and ok(img_id):
                    chosen.append(img_id)
                    chosen_set.add(img_id)
                    collected.update(boxes_per_image[img_id])
                    progress = True
                    break

    for img_id in order:
        if len(chosen) >= n:
            break
        if img_id not in chosen_set and ok(img_id):
            chosen.append(img_id)
            chosen_set.add(img_id)

    return {
        "images": [im for im in coco["images"] if im["id"] in chosen_set],
        "categories": coco["categories"],
        "annotations": [a for a in coco["annotations"] if a["image_id"] in chosen_set],
    }


def box_counts(coco: dict) -> Counter:
    """Number of boxes per category name."""
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    return Counter(cat_name[a["category_id"]] for a in coco["annotations"])


def coco_to_yolo(coco: dict, class_names: list[str]) -> dict[str, list[str]]:
    """
    YOLO label lines per image file: 'class cx cy w h', all 0..1 relative to
    the image size, class = index in class_names. Boxes are clipped to the
    image; boxes of other classes or with no area left are skipped. Every
    image gets an entry (an empty list means "no objects", which YOLO needs).
    """
    index = {n: i for i, n in enumerate(class_names)}
    cat_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    out: dict[str, list[str]] = {im["file_name"]: [] for im in coco["images"]}
    for a in coco["annotations"]:
        cls = index.get(cat_name[a["category_id"]])
        if cls is None:
            continue
        im = images[a["image_id"]]
        W, H = im["width"], im["height"]
        x, y, w, h = a["bbox"]
        x1, y1 = max(0.0, x), max(0.0, y)
        x2, y2 = min(float(W), x + w), min(float(H), y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        out[im["file_name"]].append(f"{cls} {cx:.6f} {cy:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return out
