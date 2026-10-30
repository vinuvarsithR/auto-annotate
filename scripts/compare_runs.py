"""
compare_runs.py - put several scored runs side by side, one table per class.

    python scripts/compare_runs.py runs/eot_crane_cam0 runs/A1 runs/A2 runs/A3

Reads sweep.csv and thresholds.json from each run folder (written by
run_owlv2.py in gold mode). Shows recall at fixed thresholds, plus the
per-class threshold picked for the run's --target-recall.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def fmt(v) -> str:
    return "  -  " if v is None else f"{v:5.2f}"


def read_sweep(path: Path) -> dict[tuple[float, str], dict]:
    """{(threshold, class): row}, with empty cells turned back into None."""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            out[(float(r["threshold"]), r["class"])] = {
                "recall": float(r["recall"]) if r["recall"] else None,
                "precision": float(r["precision"]) if r["precision"] else None,
                "n_pred": int(r["n_pred"]),
            }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="run folders")
    ap.add_argument("--at", type=float, nargs="+", default=[0.10, 0.20], help="thresholds to show recall at")
    args = ap.parse_args()

    runs = []
    for d in args.runs:
        if not (d / "sweep.csv").exists() or not (d / "thresholds.json").exists():
            print(f"{d}: missing sweep.csv or thresholds.json (score it with run_owlv2.py --gold first)")
            return 1
        runs.append((d.name, read_sweep(d / "sweep.csv"), json.loads((d / "thresholds.json").read_text())))

    classes = sorted({c for _, _, th in runs for c in th["classes"]})
    targets = {th["target_recall"] for _, _, th in runs}
    target = f"{targets.pop():.2f}" if len(targets) == 1 else "per run"
    width = max(len(name) for name, _, _ in runs) + 2

    for cls in classes:
        print(f"\n{cls}")
        head = "".join(f"  R@{t:.2f}" for t in args.at)
        print(f"  {'run':<{width}}{head}   | threshold for R>={target}: thresh  P      R     met")
        for name, rows, th in runs:
            cells = "".join(f"   {fmt(rows.get((t, cls), {}).get('recall'))}" for t in args.at)
            c = th["classes"].get(cls)
            if c is None or c["threshold"] is None:
                chosen = "  -"
            else:
                chosen = (f"{c['threshold']:5.2f}  {fmt(c['precision'])}  {fmt(c['recall'])}  "
                          f"{'yes' if c['met'] else 'NO'}")
            print(f"  {name:<{width}}{cells}   |                            {chosen}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
