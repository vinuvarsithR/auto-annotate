"""
pick_examples.py - choose example boxes for image-guided OWLv2 (step B),
from the TRAIN split only, and draw them on one contact sheet to check by eye.

    python scripts/pick_examples.py

Writes:  experiments/image_queries/B1.yaml   (classes.yaml + 'examples' for rb, sb, biax)
         runs/B/examples_sheet.jpg           (one tile per example, box drawn in red)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from autolabel.sequences import sequence_of  # noqa: E402


def pick(coco: dict, cls: str, k: int, seed: int, margin: int = 5) -> list[tuple[str, list[float]]]:
    """k boxes of one class: not touching the image edge, typical size (40-90th percentile area), different videos first."""
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}
    boxes = []
    for a in coco["annotations"]:
        if cat[a["category_id"]] != cls:
            continue
        im = images[a["image_id"]]
        x, y, w, h = a["bbox"]
        if x < margin or y < margin or x + w > im["width"] - margin or y + h > im["height"] - margin:
            continue
        boxes.append((w * h, im["file_name"], [round(v, 1) for v in a["bbox"]]))
    boxes.sort()
    typical = boxes[int(0.4 * len(boxes)): int(0.9 * len(boxes))] or boxes
    random.Random(seed).shuffle(typical)
    by_seq = defaultdict(list)
    for _, name, bbox in typical:
        by_seq[sequence_of(name)].append((name, bbox))
    chosen = []
    while len(chosen) < k and any(by_seq.values()):
        for seq in sorted(by_seq):
            if by_seq[seq] and len(chosen) < k:
                chosen.append(by_seq[seq].pop(0))
    return chosen


def tile(path: Path, bbox: list[float], label: str, height: int = 240) -> Image.Image:
    x, y, w, h = bbox
    with Image.open(path) as im:
        im = im.convert("RGB")
        pad = 0.4 * max(w, h)
        box = (max(0, x - pad), max(0, y - pad), min(im.width, x + w + pad), min(im.height, y + h + pad))
        crop = im.crop(box)
    d = ImageDraw.Draw(crop)
    d.rectangle([x - box[0], y - box[1], x - box[0] + w, y - box[1] + h], outline=(255, 0, 0), width=3)
    scale = height / crop.height
    crop = crop.resize((max(1, int(crop.width * scale)), height))
    out = Image.new("RGB", (max(crop.width, 260), height + 22), "white")
    out.paste(crop, (0, 22))
    ImageDraw.Draw(out).text((4, 4), label, fill=(0, 0, 0))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=ROOT / "data/splits/eot_crane_v1")
    ap.add_argument("--classes", type=Path, default=ROOT / "gold_sets/eot_crane_test50/classes.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/image_queries/B1.yaml")
    ap.add_argument("--sheet", type=Path, default=ROOT / "runs/B/examples_sheet.jpg")
    ap.add_argument("--image-classes", nargs="+", default=["rb", "sb", "biax"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    info = json.loads((args.split / "split_info.json").read_text())
    src = {p.name: p for p in (ROOT / info["export"]).rglob("*.jpg")}
    train = json.loads((args.split / "train.json").read_text())

    data = yaml.safe_load(args.classes.read_text())
    tiles = []
    for c in data["classes"]:
        if c["name"] not in args.image_classes:
            continue
        chosen = pick(train, c["name"], args.k, args.seed)
        c["examples"] = [
            {"image": os.path.relpath(src[name], args.out.parent), "bbox": bbox} for name, bbox in chosen
        ]
        for i, (name, bbox) in enumerate(chosen, 1):
            print(f"  {c['name']} #{i}: {name}  bbox {bbox}")
            tiles.append(tile(src[name], bbox, f"{c['name']} #{i}  {sequence_of(name)}"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = "# Step B: image-guided queries. Examples picked from the TRAIN split by scripts/pick_examples.py.\n"
    args.out.write_text(header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True))

    cols = args.k
    rows = (len(tiles) + cols - 1) // cols
    cw, ch = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new("RGB", (cw * cols, ch * rows), "white")
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * cw, (i // cols) * ch))
    args.sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.sheet, quality=90)
    print(f"Wrote {args.out} and {args.sheet}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
