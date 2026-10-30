"""
make_split.py - cut the Roboflow export into train / val / test by time
inside each video (see autolabel/split.py), with no leaks.

    python scripts/make_split.py

Reads:   data/roboflow/eot_crane-yduh1-vhu5p-ohmpg-v1/   (the v1 export, already on disk)
Writes:  gold_sets/eot_crane_test50/   images/, annotations.json, classes.yaml   (PROVISIONAL labels)
         data/splits/eot_crane_v1/     train.json, val.json (COCO), split_info.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from autolabel.coco import merge_coco  # noqa: E402
from autolabel.dedup import dhash  # noqa: E402
from autolabel.gold import alias_map, load_classes  # noqa: E402
from autolabel.sequences import sequence_of  # noqa: E402
from autolabel.split import TEST, TRAIN, VAL, Windows, build_split, check_no_leak  # noqa: E402


def canonical(coco: dict, names: dict[str, str]) -> tuple[dict, Counter]:
    """Keep only our classes (via classes.yaml aliases); categories become the canonical names."""
    classes = sorted(set(names.values()))
    new_id = {c: i for i, c in enumerate(classes)}
    old = {c["id"]: c["name"] for c in coco["categories"]}
    anns, skipped = [], Counter()
    for a in coco["annotations"]:
        c = names.get(old[a["category_id"]].strip().lower())
        if c is None:
            skipped[old[a["category_id"]]] += 1
            continue
        anns.append({**a, "category_id": new_id[c]})
    cats = [{"id": i, "name": c} for c, i in new_id.items()]
    return {"images": coco["images"], "categories": cats, "annotations": anns}, skipped


def counts(coco: dict) -> tuple[int, Counter]:
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    return len(coco["images"]), Counter(cat[a["category_id"]] for a in coco["annotations"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", type=Path, default=ROOT / "data/roboflow/eot_crane-yduh1-vhu5p-ohmpg-v1")
    ap.add_argument("--classes", type=Path, default=ROOT / "gold_sets/eot_crane_cam0/classes.yaml")
    ap.add_argument("--test-name", default="eot_crane_test50")
    ap.add_argument("--split-name", default="eot_crane_v1")
    ap.add_argument("--n-test", type=int, default=50)
    ap.add_argument("--max-per-seq", type=int, default=10)
    ap.add_argument("--min-test-gap", type=int, default=5, help="frames between two test images of one video")
    ap.add_argument("--max-dist", type=int, default=16, help="dHash bits: <= this counts as 'looks alike'")
    ap.add_argument("--test-frac", type=float, default=0.30, help="last part of each video used for test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    test_dir = ROOT / "gold_sets" / args.test_name
    split_dir = ROOT / "data" / "splits" / args.split_name
    if (test_dir / "annotations.json").exists():
        sys.exit(f"{test_dir} already exists. Delete it first to rebuild.")

    parts, src = [], {}
    for jf in sorted(args.export.rglob("_annotations.coco.json")):
        parts.append(json.loads(jf.read_text()))
        for p in jf.parent.glob("*.jpg"):
            src[p.name] = p
    coco, skipped = canonical(merge_coco(parts), alias_map(load_classes(args.classes)))
    print(f"{len(coco['images'])} images; skipped categories: {dict(skipped)}")

    print("Hashing images...")
    hashes = {}
    for im in coco["images"]:
        with Image.open(src[im["file_name"]]) as pil:
            hashes[im["file_name"]] = dhash(pil)

    w = Windows(test=args.test_frac)
    r = build_split(coco, hashes, args.n_test, w, args.max_per_seq, args.min_test_gap, args.max_dist, args.seed)
    split_of = {im["file_name"]: s for s in (TRAIN, VAL, TEST) for im in r[s]["images"]}
    problems = check_no_leak(split_of, w.min_gap_frames, hashes, args.max_dist)
    if problems:
        print("LEAK CHECK FAILED:")
        for p in problems[:20]:
            print("  " + p)
        return 1
    print("Leak check: OK (time gaps and look-alikes)")

    # Write test gold set
    (test_dir / "images").mkdir(parents=True)
    for im in r[TEST]["images"]:
        shutil.copy2(src[im["file_name"]], test_dir / "images" / im["file_name"])
    test = dict(r[TEST])
    test["info"] = {
        "description": "PROVISIONAL gold set: copied from Roboflow labels, not yet re-checked by hand.",
        "source": f"{args.export.name}, test window of make_split.py, seed {args.seed}",
    }
    (test_dir / "annotations.json").write_text(json.dumps(test, indent=1))
    shutil.copy2(args.classes, test_dir / "classes.yaml")

    # Write train/val lists (COCO, images stay in the export folder)
    split_dir.mkdir(parents=True, exist_ok=True)
    for s in (TRAIN, VAL):
        (split_dir / f"{s}.json").write_text(json.dumps(r[s]))
    info = {
        "export": str(args.export.relative_to(ROOT)),
        "windows": vars(w),
        "params": {k: v for k, v in vars(args).items() if k not in ("export", "classes")},
        "dropped_lookalike": r["dropped_lookalike"],
        "window_counts": Counter(r["windows"].values()),
    }
    (split_dir / "split_info.json").write_text(json.dumps(info, indent=1, default=str))

    # Report
    classes = [c["name"] for c in coco["categories"]]
    print(f"\nImages and boxes per split (windows: {dict(Counter(r['windows'].values()))})")
    print(f"  {'split':<6} {'images':>6} " + " ".join(f"{c:>7}" for c in classes))
    for s in (TRAIN, VAL, TEST):
        n, c = counts(r[s])
        print(f"  {s:<6} {n:>6} " + " ".join(f"{c[k]:>7}" for k in classes))
    print(f"  dropped as look-alikes of test images: {len(r['dropped_lookalike'])}")

    print("\nImages per video and split")
    table = defaultdict(Counter)
    for name, s in split_of.items():
        table[sequence_of(name)][s] += 1
    for seq in sorted(table):
        t = table[seq]
        print(f"  {seq:<40} train {t[TRAIN]:>4}  val {t[VAL]:>3}  test {t[TEST]:>3}")

    print("\nTest boxes per video and class")
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    img = {im["id"]: im["file_name"] for im in r[TEST]["images"]}
    tb = defaultdict(Counter)
    for a in r[TEST]["annotations"]:
        tb[sequence_of(img[a["image_id"]])][cat[a["category_id"]]] += 1
    for seq in sorted(tb):
        print(f"  {seq:<40} " + "  ".join(f"{k} {tb[seq][k]}" for k in classes))

    print(f"\nWrote {test_dir} and {split_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
