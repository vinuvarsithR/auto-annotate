"""
compare_models.py - score every model in runs/compare/ on the same test images.

    python scripts/compare_models.py

For each model:
  1. Per-class thresholds picked on VAL: highest threshold keeping recall >= --target-recall.
  2. Those thresholds applied to TEST: precision, recall, class-ignored recall.
  3. mAP@0.5 on TEST from all saved boxes (doesn't depend on thresholds, so
     models with different score scales can be compared).
For owl_image also 'owl_image (sb/biax keep higher)': where an sb and a biax
box overlap (IoU >= 0.5), only the higher-scoring one is kept.

Writes runs/compare/results.json, results.md and report.html.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from autolabel.evaluate import ALL, Box, evaluate, map50, sweep  # noqa: E402
from autolabel.gold import load_gold_set  # noqa: E402
from autolabel.merge import keep_higher  # noqa: E402
from autolabel.report import build_report  # noqa: E402
from autolabel.thresholds import apply_thresholds, select_thresholds  # noqa: E402

TEST_DIR = ROOT / "gold_sets/eot_crane_test50"
SPLIT_DIR = ROOT / "data/splits/eot_crane_v1"
RUNS = ROOT / "runs/compare"
GRID = [round(0.01 * i, 2) for i in range(1, 100)]
MODELS = [("owl_text", "OWLv2 text"), ("owl_image", "OWLv2 image-guided (B)"), ("gdino", "Grounding DINO"),
          ("yolo", "YOLO (trained)")]
KEEP_HIGHER = "OWLv2 image-guided (B), sb/biax keep higher"


def load_preds(path: Path) -> list[Box]:
    return [Box(p["file_name"], p["category"], tuple(p["bbox"]), p["score"]) for p in json.loads(path.read_text())]


def val_gold() -> list[Box]:
    coco = json.loads((SPLIT_DIR / "val.json").read_text())
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    img = {im["id"]: im["file_name"] for im in coco["images"]}
    return [Box(img[a["image_id"]], cat[a["category_id"]], tuple(float(v) for v in a["bbox"])) for a in coco["annotations"]]


def f(v) -> str:
    return "-" if v is None else f"{v:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    test = load_gold_set(TEST_DIR)
    classes = [c.name for c in test.classes]
    vgold = val_gold()

    runs = {}
    for key, label in MODELS:
        if (RUNS / key / "test.json").exists():
            runs[label] = (load_preds(RUNS / key / "val.json"), load_preds(RUNS / key / "test.json"),
                           json.loads((RUNS / key / "meta.json").read_text()))
        else:
            print(f"(no run for {key}, skipped)")
    owl_label = dict(MODELS)["owl_image"]
    if owl_label in runs:
        v, t, m = runs[owl_label]
        runs[KEEP_HIGHER] = (keep_higher(v, {"sb", "biax"}, args.iou), keep_higher(t, {"sb", "biax"}, args.iou), m)

    results, shown = {}, {}
    for label, (vpreds, tpreds, meta) in runs.items():
        chosen = select_thresholds(sweep(vpreds, vgold, GRID, args.iou), args.target_recall)
        kept = apply_thresholds(tpreds, chosen)
        scores = evaluate(kept, test.boxes, 0.0, args.iou)
        aps = map50(tpreds, test.boxes, args.iou)
        results[label] = {"thresholds": chosen, "test": scores, "ap50": aps, "meta": meta}
        shown[label] = kept

    # ---- tables (rows of cells; written as markdown and as HTML) ----
    overall = [["model", "precision", "recall", "class-ignored recall", "mAP@0.5", "img/s (test)", "peak VRAM GB"]]
    for label, r in results.items():
        o, m = r["test"][ALL], r["meta"]
        overall.append([label, f(o["precision"]), f(o["recall"]), f(o["recall_class_ignored"]), f(r["ap50"][ALL]),
                        str(m.get("test_img_per_s", "-")), str(m.get("peak_vram_gb", "-"))])
    per_class = [["model", "class", "gold", "threshold (val)", "precision", "recall", "AP@0.5"]]
    for label, r in results.items():
        for cls in classes:
            t, s = r["thresholds"].get(cls, {}), r["test"].get(cls, {})
            met = "" if t.get("met") else " (target not met)"
            per_class.append([label, cls, str(s.get("n_gold", 0)), f(t.get("threshold")) + met,
                              f(s.get("precision")), f(s.get("recall")), f(r["ap50"].get(cls))])

    def md_table(rows) -> str:
        return "\n".join(["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
                         + ["| " + " | ".join(r) + " |" for r in rows[1:]])

    def html_table(rows) -> str:
        head = "<tr>" + "".join(f"<th>{c}</th>" for c in rows[0]) + "</tr>"
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows[1:])
        return f"<table>{head}{body}</table>"

    intro = (f"Per-class thresholds picked on VAL (recall target {args.target_recall:.2f}), scored on TEST "
             f"({len(test.image_files)} images). mAP@0.5 uses all saved boxes (threshold-free).")
    md = "\n\n".join([intro, md_table(overall), md_table(per_class)])
    print(md)
    (RUNS / "results.md").write_text(md + "\n")
    (RUNS / "results.json").write_text(json.dumps(results, indent=1, default=str))

    # ---- HTML report ----
    table = f"<p>{intro}</p>" + html_table(overall) + html_table(per_class)

    notes = [
        "<b>PROVISIONAL labels:</b> gold boxes are copied from Roboflow and have NOT been hand-checked. "
        "Green dashed boxes below are the gold; use this page to eyeball them.",
        "<b>Optimistic split:</b> only 8 videos from one camera on one day exist. Test frames come from the last 30% "
        "of each video, with a gap of at least 15 frames to any train/val frame and no look-alike images "
        "(256-bit dHash, &le;16 bits). Scores on new crane jobs will likely be lower.",
        f"<b>sb has only {sum(1 for b in test.boxes if b.category == 'sb')} test boxes</b> (one video): its numbers are very noisy.",
        "<b>Thresholds</b> per model and class were picked on the VAL split (highest threshold keeping recall &ge; "
        f"{args.target_recall:.2f}), then applied to test. Boxes shown are at those thresholds.",
        "<b>Licence:</b> the YOLO model uses Ultralytics (AGPL-3.0). It was used here for an internal experiment only; "
        "the team still has to decide whether AGPL-3.0 is acceptable for any product use.",
    ]
    gold = {}
    for b in test.boxes:
        gold.setdefault(b.image_id, []).append((b.category, b.bbox))
    models = {}
    for label, kept in shown.items():
        d = {}
        for p in kept:
            d.setdefault(p.image_id, []).append((p.category, p.bbox, p.score))
        models[label] = d
    images = [(n, TEST_DIR / "images" / n) for n in test.image_files]
    build_report(RUNS / "report.html", "EOT crane: detector comparison", notes, table, images, gold, models)
    print(f"\nReport: {RUNS / 'report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
