"""
run_owlv2.py - Phase 1: auto-label images with OWLv2 and score against a gold set.

Typical use (score against a gold set):
    python scripts/run_owlv2.py --gold gold_sets/eot_crane_provisional

Just label a folder (no scoring):
    python scripts/run_owlv2.py --images path/to/images --classes path/to/classes.yaml

Try other prompts on the same gold set (class names must match the gold set's):
    python scripts/run_owlv2.py --gold gold_sets/eot_crane_provisional \\
        --classes experiments/person_prompts/A1.yaml --out runs/A1

Re-score saved predictions without running the model again:
    python scripts/run_owlv2.py --gold gold_sets/eot_crane_provisional --score-only

Outputs go to runs/<name>/:
    predictions.json   every box with score >= --save-threshold (default 0.05)
    sweep.csv          precision/recall per class at each threshold (gold mode only)
    thresholds.json    per-class threshold chosen for --target-recall (gold mode only)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autolabel.evaluate import ALL, DEFAULT_THRESHOLDS, Box, sweep  # noqa: E402

CSV_FIELDS = ["threshold", "class", "tp", "fp", "fn", "n_gold", "n_pred",
              "precision", "recall", "recall_class_ignored"]
from autolabel.gold import list_images, load_classes, load_gold_set  # noqa: E402
from autolabel.thresholds import select_thresholds  # noqa: E402


def fmt(v) -> str:
    return "  -  " if v is None else f"{v:5.2f}"


def run_model(image_paths, classes, out_file, model_id, device, save_threshold):
    from PIL import Image

    from autolabel.owlv2 import Owlv2Detector

    prompts = [c.description for c in classes]
    print(f"Loading {model_id} (first run downloads ~620 MB)...")
    det = Owlv2Detector(model_id, device)
    print(f"Running on: {det.device}")
    print("Prompts:")
    for c in classes:
        print(f"  {c.name:<20} '{c.description}'")

    preds = []
    start = time.time()
    for n, path in enumerate(image_paths, 1):
        with Image.open(path) as im:
            image = im.convert("RGB")
        for d in det.detect(image, prompts, save_threshold):
            preds.append(
                {
                    "file_name": path.name,
                    "category": classes[d.label].name,
                    "bbox": [round(v, 2) for v in d.bbox],
                    "score": round(d.score, 4),
                }
            )
        if n % 25 == 0 or n == len(image_paths):
            rate = n / (time.time() - start)
            print(f"  {n}/{len(image_paths)} images  ({rate:.1f} img/s)")

    vram = det.peak_vram_gb()
    if vram is not None:
        print(f"Peak GPU memory: {vram:.2f} GB")

    out_file.write_text(json.dumps(preds, indent=1))
    print(f"Saved {len(preds)} boxes to {out_file}")
    return preds


def to_boxes(preds) -> list[Box]:
    return [Box(p["file_name"], p["category"], tuple(p["bbox"]), p["score"]) for p in preds]


def print_tables(rows, classes, report_threshold):
    overall = [r for r in rows if r["class"] == ALL]
    print("\nOverall, by confidence threshold")
    print("  thresh  precision  recall  recall(class ignored)  boxes drawn")
    for r in overall:
        print(
            f"  {r['threshold']:5.2f}    {fmt(r['precision'])}    {fmt(r['recall'])}"
            f"          {fmt(r['recall_class_ignored'])}            {r['n_pred']}"
        )

    print(f"\nPer class at threshold {report_threshold:.2f}")
    print("  class                 gold boxes  precision  recall")
    at = {r["class"]: r for r in rows if abs(r["threshold"] - report_threshold) < 1e-9}
    for c in classes:
        r = at.get(c.name)
        if r is None:
            print(f"  {c.name:<20}       0        -        -   (no gold boxes, no predictions)")
            continue
        print(f"  {c.name:<20} {r['n_gold']:>8}     {fmt(r['precision'])}    {fmt(r['recall'])}")


def print_chosen(chosen, classes, target_recall):
    print(f"\nPer-class threshold: highest that keeps recall >= {target_recall:.2f}")
    print("  class                 gold boxes  threshold  precision  recall  target met")
    for c in classes:
        r = chosen.get(c.name)
        if r is None or r["threshold"] is None:
            print(f"  {c.name:<20}       0        -          -        -     (no gold boxes)")
            continue
        met = "yes" if r["met"] else "NO (best recall shown)"
        print(
            f"  {c.name:<20} {r['n_gold']:>8}      {r['threshold']:5.2f}      {fmt(r['precision'])}"
            f"    {fmt(r['recall'])}  {met}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, help="gold set folder (images/, classes.yaml, annotations.json)")
    ap.add_argument("--images", type=Path, help="image folder (when not using --gold)")
    ap.add_argument("--classes", type=Path, help="classes.yaml (with --gold: overrides the gold set's own)")
    ap.add_argument("--out", type=Path, help="output folder (default runs/<gold or images folder name>)")
    ap.add_argument("--model", default="google/owlv2-base-patch16-ensemble")
    ap.add_argument("--device", default="auto", help="auto, cuda, cuda:0 or cpu")
    ap.add_argument("--save-threshold", type=float, default=0.05, help="lowest score kept in predictions.json")
    ap.add_argument("--report-threshold", type=float, default=0.20, help="threshold for the per-class table")
    ap.add_argument("--iou", type=float, default=0.5, help="IoU needed to count a match")
    ap.add_argument("--target-recall", type=float, default=0.90,
                    help="per-class threshold = highest one keeping recall >= this")
    ap.add_argument("--score-only", action="store_true", help="reuse predictions.json, don't run the model")
    args = ap.parse_args()

    if args.gold:
        gold = load_gold_set(args.gold, args.classes)
        classes = gold.classes
        image_dir = args.gold / "images"
        name = args.gold.name
        if gold.skipped_categories:
            print("Skipped gold categories not in classes.yaml:", dict(gold.skipped_categories))
    elif args.images and args.classes:
        gold = None
        classes = load_classes(args.classes)
        image_dir = args.images
        name = args.images.name
    else:
        ap.error("give --gold, or both --images and --classes")

    out_dir = args.out or ROOT / "runs" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_file = out_dir / "predictions.json"

    if args.score_only:
        if not pred_file.exists():
            ap.error(f"--score-only but {pred_file} doesn't exist yet")
        preds = json.loads(pred_file.read_text())
        print(f"Loaded {len(preds)} saved boxes from {pred_file}")
    else:
        images = list_images(image_dir)
        if gold:
            wanted = set(gold.image_files)
            missing = wanted - {p.name for p in images}
            if missing:
                print(f"WARNING: {len(missing)} images in annotations.json are not in {image_dir}")
            images = [p for p in images if p.name in wanted]
        if not images:
            print(f"No images found in {image_dir}")
            return 1
        print(f"{len(images)} images, {len(classes)} classes")
        preds = run_model(images, classes, pred_file, args.model, args.device, args.save_threshold)

    if gold is None:
        return 0

    thresholds = [t for t in DEFAULT_THRESHOLDS if t >= args.save_threshold]
    if args.report_threshold not in thresholds:
        thresholds = sorted(set(thresholds) | {args.report_threshold})
    rows = sweep(to_boxes(preds), gold.boxes, thresholds, args.iou)

    with open(out_dir / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    print_tables(rows, classes, args.report_threshold)

    chosen = select_thresholds(rows, args.target_recall)
    print_chosen(chosen, classes, args.target_recall)
    (out_dir / "thresholds.json").write_text(
        json.dumps({"target_recall": args.target_recall, "iou": args.iou, "classes": chosen}, indent=1)
    )
    print(f"\nFull table: {out_dir / 'sweep.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
