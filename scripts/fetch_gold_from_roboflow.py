"""
fetch_gold_from_roboflow.py - build a PROVISIONAL gold set from existing
Roboflow labels.

"Provisional" means the boxes have not been re-checked by hand yet, so scores
measured against it are approximate. Re-check before trusting the numbers.

Before running:
  1. Put ROBOFLOW_API_KEY in .env (see .env.example). The key is read from
     there and never printed.
  2. In Roboflow, generate a dataset version for the project with
     NO augmentation and NO resize (Preprocessing: remove "Resize").
     A resize would stretch the images OWLv2 sees.

Example:
    python scripts/fetch_gold_from_roboflow.py \\
        --workspace vinus-workspace-4ji2t --project eot_crane --version 1 \\
        --name eot_crane_provisional --n 50
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autolabel.coco import box_counts, merge_coco, sample_coco  # noqa: E402


def read_api_key() -> str:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not key:
        sys.exit("ROBOFLOW_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workspace", required=True, help="workspace slug from the Roboflow URL")
    ap.add_argument("--project", required=True, help="project slug from the Roboflow URL")
    ap.add_argument("--version", type=int, required=True, help="dataset version number")
    ap.add_argument("--name", required=True, help="gold set folder name, e.g. eot_crane_provisional")
    ap.add_argument("--n", type=int, default=50, help="number of images to pick")
    ap.add_argument("--per-class", type=int, default=50, help="try to reach this many boxes per class")
    ap.add_argument("--seed", type=int, default=0, help="change to get a different random pick")
    args = ap.parse_args()

    out = ROOT / "gold_sets" / args.name
    if (out / "annotations.json").exists():
        sys.exit(f"{out} already has annotations.json. Pick another --name or delete it first.")

    from roboflow import Roboflow

    download_dir = ROOT / "data" / "roboflow" / f"{args.project}-v{args.version}"
    rf = Roboflow(api_key=read_api_key())
    version = rf.workspace(args.workspace).project(args.project).version(args.version)
    version.download("coco", location=str(download_dir), overwrite=True)

    # Roboflow puts one COCO file per split next to that split's images.
    parts, image_dirs = [], []
    for jf in sorted(download_dir.rglob("*.json")):
        data = json.loads(jf.read_text())
        if isinstance(data, dict) and "images" in data and "annotations" in data:
            parts.append(data)
            image_dirs.append(jf.parent)
    if not parts:
        sys.exit(f"No COCO files found in {download_dir}")

    merged = merge_coco(parts)
    picked = sample_coco(merged, args.n, args.per_class, args.seed)

    (out / "images").mkdir(parents=True, exist_ok=True)
    copied = 0
    for im in picked["images"]:
        for d in image_dirs:
            src = d / im["file_name"]
            if src.exists():
                shutil.copy2(src, out / "images" / im["file_name"])
                copied += 1
                break
        else:
            print(f"WARNING: image file not found: {im['file_name']}")

    picked["info"] = {
        "description": "PROVISIONAL gold set: copied from Roboflow labels, not yet re-checked by hand.",
        "source": f"{args.workspace}/{args.project} v{args.version}, seed {args.seed}",
    }
    (out / "annotations.json").write_text(json.dumps(picked, indent=1))

    print(f"\nGold set written to {out}")
    print(f"  {copied} images (from {len(merged['images'])} in the export)")
    print("  boxes per class:")
    for name, count in box_counts(picked).most_common():
        print(f"    {name:<24} {count}")
    if not (out / "classes.yaml").exists():
        print("\nNext: add classes.yaml (copy gold_sets/_template/classes.yaml and fill in descriptions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
