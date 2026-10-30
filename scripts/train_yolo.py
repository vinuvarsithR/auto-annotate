"""
train_yolo.py - train a small YOLO on the train split from make_split.py.

    python scripts/train_yolo.py

LICENCE: Ultralytics is AGPL-3.0. Used here for an internal experiment only;
the team decides later whether that licence is acceptable for production.

Builds data/yolo/<split>/ (images are symlinks into the Roboflow export,
labels are YOLO .txt files), re-checks that no train/val image leaks into
the test set, then trains. Weights end up in runs/yolo/<name>/weights/best.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autolabel.coco import coco_to_yolo  # noqa: E402
from autolabel.gold import load_classes  # noqa: E402
from autolabel.split import TEST, TRAIN, VAL, Windows, check_no_leak  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", type=Path, default=ROOT / "data/splits/eot_crane_v1")
    ap.add_argument("--test", type=Path, default=ROOT / "gold_sets/eot_crane_test50")
    ap.add_argument("--model", default="yolo26s.pt", help="Ultralytics weights to start from")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="eot_crane_v1_yolo26s")
    args = ap.parse_args()

    info = json.loads((args.split / "split_info.json").read_text())
    export = ROOT / info["export"]
    class_names = [c.name for c in load_classes(args.test / "classes.yaml")]
    splits = {s: json.loads((args.split / f"{s}.json").read_text()) for s in (TRAIN, VAL)}
    test_files = [im["file_name"] for im in json.loads((args.test / "annotations.json").read_text())["images"]]

    # Leak check again, right before training.
    split_of = {n: TEST for n in test_files}
    for s, coco in splits.items():
        for im in coco["images"]:
            if im["file_name"] in split_of:
                sys.exit(f"LEAK: {im['file_name']} is in {s} and in {split_of[im['file_name']]}")
            split_of[im["file_name"]] = s
    problems = check_no_leak(split_of, Windows(**info["windows"]).min_gap_frames)
    if problems:
        sys.exit("LEAK CHECK FAILED:\n  " + "\n  ".join(problems[:20]))
    print("Leak check: OK")

    src = {p.name: p for p in export.rglob("*.jpg")}
    yolo_dir = ROOT / "data" / "yolo" / args.split.name
    for s, coco in splits.items():
        (yolo_dir / "images" / s).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
        for name, lines in coco_to_yolo(coco, class_names).items():
            link = yolo_dir / "images" / s / name
            if not link.exists():
                link.symlink_to(src[name].resolve())
            (yolo_dir / "labels" / s / (Path(name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
    data_yaml = yolo_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {yolo_dir}\ntrain: images/train\nval: images/val\n"
        + "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names))
    )
    print(f"YOLO dataset: {yolo_dir} ({len(splits[TRAIN]['images'])} train, {len(splits[VAL]['images'])} val)")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml), imgsz=args.imgsz, epochs=args.epochs, patience=args.patience, batch=args.batch,
        project=str(ROOT / "runs" / "yolo"), name=args.name, exist_ok=True, seed=0, deterministic=True,
        workers=4, plots=True,
    )
    print(f"Best weights: {ROOT / 'runs' / 'yolo' / args.name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
