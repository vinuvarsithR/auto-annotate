"""
train_package.py - the Colab training package zip (CONTRACT #23, "Train on Colab").

Zip layout
----------
    data.yaml                       path /content/dataset, train/val folders, nc, names
    split.json                      which sequences went to train / val, and why (see below)
    train.ipynb                     4-cell Colab notebook: intro, upload, train, download
    images/train/<name>   labels/train/<stem>.txt
    images/val/<name>     labels/val/<stem>.txt     (absent when there is only one sequence)

Images and label lines are exactly those of #14 (`yolo` for detect, `yolo-seg` for segment),
built with the same functions. Train / val is split BY SEQUENCE (one video never lands in both),
because neighbouring frames look almost the same and would make val look better than it is.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import PurePosixPath
from typing import Callable

from autolabel.export.formats import export_names, to_yolo
from autolabel.export.select import (
    APPROVED_STATUSES,
    NothingToExport,
    boxes_for_image,
    exportable_images,
)
from autolabel.sequences import sequence_of

TRAIN_TASKS = ("detect", "segment")
VAL_TARGET = 0.2
# Must match the ultralytics pin in requirements.txt (a contract test checks this).
ULTRALYTICS_PIN = "ultralytics==8.4.199"
BASE_WEIGHTS = {"detect": "yolo11s.pt", "segment": "yolo11s-seg.pt"}
IMGSZ = 960
EPOCHS = 100


class NoPolygons(Exception):
    """segment was asked but no exported box has a polygon (the route answers 409)."""


def safe_name(name: str) -> str:
    """Project name made safe for file names (same rule as the #14 zip name)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "project"


# ------------------------------------------------------------------ split rule

def split_sequences(image_seq: dict[int, str], image_classes: dict[int, set[int]],
                    ) -> tuple[list[str], list[str], bool]:
    """
    image id -> sequence, image id -> class indexes of its exported boxes.
    Returns (train sequences, val sequences, val_is_train), both lists sorted by name.
    """
    seq_images: dict[str, list[int]] = {}
    for iid, seq in image_seq.items():
        seq_images.setdefault(seq, []).append(iid)
    seq_classes = {s: set().union(*(image_classes.get(i, set()) for i in ids))
                   for s, ids in seq_images.items()}
    all_seqs = sorted(seq_images)
    if len(all_seqs) == 1:
        return all_seqs, [], True

    n = len(image_seq)
    target = max(1, round(VAL_TARGET * n))
    order = sorted(all_seqs, key=lambda s: (len(seq_images[s]), s))
    needed = set().union(*seq_classes.values())

    def train_ok(val: set[str]) -> bool:          # rule (b)
        return len(val) < len(all_seqs)

    def classes_ok(val: set[str]) -> bool:        # rule (c)
        kept = set().union(*(seq_classes[s] for s in all_seqs if s not in val))
        return needed <= kept

    val: set[str] = set()
    val_images = 0
    for s in order:
        cand = val | {s}
        if val_images + len(seq_images[s]) <= target and train_ok(cand) and classes_ok(cand):
            val, val_images = cand, val_images + len(seq_images[s])
    if not val:
        pick = next((s for s in order if train_ok({s}) and classes_ok({s})), None)
        if pick is None:
            pick = next(s for s in order if train_ok({s}))
        val = {pick}
    return sorted(s for s in all_seqs if s not in val), sorted(val), False


# ------------------------------------------------------------------ notebook

def _cell(kind: str, text: str) -> dict:
    cell = {"cell_type": kind, "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        cell.update(outputs=[], execution_count=None)
    return cell


def notebook(project_name: str, task: str, split: dict) -> dict:
    """The 4-cell Colab notebook of #23 as a JSON-ready dict."""
    run = f"{safe_name(project_name)}_{task}"
    if split["val_is_train"]:
        counts = (f"{split['train']['images']} images, one sequence only: "
                  "validation uses the training images.")
    else:
        counts = (f"train {split['train']['images']} images "
                  f"({len(split['train']['sequences'])} sequences), "
                  f"val {split['val']['images']} images "
                  f"({len(split['val']['sequences'])} sequences).")
    intro = (
        f"# Train {task} model: {project_name}\n"
        "\n"
        f"Task: **{task}**. Split by sequence: {counts}\n"
        "\n"
        "First set the GPU: **Runtime > Change runtime type > GPU**.\n"
        "\n"
        "1. Run the **upload** cell and pick the training package zip.\n"
        "2. Run the **train** cell and wait until it finishes.\n"
        f"3. Run the **download** cell; it saves `{run}_best.pt`. Upload that file in the app.\n"
    )
    upload = (
        f"!pip install -q {ULTRALYTICS_PIN}\n"
        "import shutil, zipfile\n"
        "from google.colab import files\n"
        "uploaded = files.upload()\n"
        "zip_name = next(iter(uploaded))\n"
        'shutil.rmtree("/content/dataset", ignore_errors=True)\n'
        'zipfile.ZipFile(zip_name).extractall("/content/dataset")\n'
        'print("dataset ready in /content/dataset")\n'
    )
    train = (
        "from ultralytics import YOLO\n"
        f'model = YOLO("{BASE_WEIGHTS[task]}")\n'
        f'model.train(data="/content/dataset/data.yaml", imgsz={IMGSZ}, epochs={EPOCHS}, '
        f'project="/content/runs", name="{run}", exist_ok=True)\n'
    )
    download = (
        "import shutil\n"
        "from google.colab import files\n"
        f'shutil.copy("/content/runs/{run}/weights/best.pt", "/content/{run}_best.pt")\n'
        f'files.download("/content/{run}_best.pt")\n'
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "colab": {"provenance": []},
        },
        "cells": [_cell("markdown", intro), _cell("code", upload),
                  _cell("code", train), _cell("code", download)],
    }


# ------------------------------------------------------------------ package

def build_train_package(project_name: str, task: str,
                        images: list[dict], classes: list[dict], annotations: list[dict],
                        image_bytes: Callable[[dict], bytes]) -> bytes:
    """
    API JSON shapes in (as build_zip); returns the #23 zip. Raises ValueError (bad task),
    NothingToExport (no done/empty image), NoPolygons. Calls image_bytes only for exported
    images, and only after every check passed.
    """
    if task not in TRAIN_TASKS:
        raise ValueError(f"task must be one of {list(TRAIN_TASKS)}")
    chosen = exportable_images(images)
    if not chosen:
        raise NothingToExport("no image has review_status done or empty")
    statuses = list(APPROVED_STATUSES)
    if task == "segment" and not any(
            a.get("polygon") for im in chosen for a in boxes_for_image(im, annotations, statuses)):
        raise NoPolygons("no exported box has a polygon")

    names = export_names(chosen)
    files = to_yolo(chosen, classes, annotations, statuses, seg=(task == "segment"))
    labels = {im["id"]: files[f"labels/{PurePosixPath(names[im['id']]).stem}.txt"]
              for im in chosen}
    # exported box = a label line actually written (a box clipped to nothing is left out)
    image_classes = {iid: {int(line.split()[0]) for line in text.splitlines() if line.strip()}
                     for iid, text in labels.items()}
    image_seq = {im["id"]: sequence_of(im["file_name"]) for im in chosen}

    train_seqs, val_seqs, val_is_train = split_sequences(image_seq, image_classes)
    part_of = {iid: ("val" if seq in val_seqs else "train") for iid, seq in image_seq.items()}

    class_names = [c["name"] for c in classes]
    def classes_in(part: str) -> set[int]:
        return set().union(*(c for iid, c in image_classes.items() if part_of[iid] == part))
    with_boxes = set().union(*image_classes.values())
    missing_train = [class_names[i] for i in range(len(classes))
                     if i in with_boxes and i not in classes_in("train")]
    missing_val = [] if val_is_train else [class_names[i] for i in range(len(classes))
                                           if i in with_boxes and i not in classes_in("val")]
    split = {
        "rule": "by_sequence",
        "val_target": VAL_TARGET,
        "val_is_train": val_is_train,
        "train": {"sequences": train_seqs,
                  "images": sum(1 for p in part_of.values() if p == "train")},
        "val": {"sequences": val_seqs, "images": sum(1 for p in part_of.values() if p == "val")},
        "classes_missing_in_train": missing_train,
        "classes_missing_in_val": missing_val,
    }
    data_yaml = "\n".join([
        "path: /content/dataset",
        "train: images/train",
        "val: " + ("images/train" if val_is_train else "images/val"),
        f"nc: {len(class_names)}",
        "names: " + json.dumps(class_names),
        "",
    ])

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", data_yaml)
        zf.writestr("split.json", json.dumps(split, indent=1))
        zf.writestr("train.ipynb", json.dumps(notebook(project_name, task, split), indent=1))
        for im in chosen:
            part, name = part_of[im["id"]], names[im["id"]]
            zf.writestr(f"labels/{part}/{PurePosixPath(name).stem}.txt", labels[im["id"]])
            zf.writestr(f"images/{part}/{name}", image_bytes(im))
    return buf.getvalue()
