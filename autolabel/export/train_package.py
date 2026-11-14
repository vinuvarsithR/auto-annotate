"""
train_package.py - the Colab training package zip (CONTRACT #23, "Train on Colab").

Zip layout
----------
    data.yaml                       path /content/dataset, train/val folders, nc, names
    split.json                      which sequences went to train / val, and why (see below)
    train.ipynb                     4-cell Colab notebook: intro, upload, train, val + download
    images/train/<name>   labels/train/<stem>.txt
    images/val/<name>     labels/val/<stem>.txt     (absent when there is only one sequence)

Images and label lines are exactly those of #14 (`yolo` for detect, `yolo-seg` for segment),
built with the same functions. Train / val is split BY SEQUENCE (one video never lands in both),
because neighbouring frames look almost the same and would make val look better than it is.

The notebook's last cell writes `<run>_metrics.json` (final val mAP + the split); the app reads it
back with `read_metrics` when the weights are imported (CONTRACT "Teammates + training routes").
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Any, Callable

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
EPOCHS_RANGE = (1, 500)
IMGSZ_RANGE = (320, 2048)          # and a multiple of 32
ALLOWED_WEIGHTS = {
    "detect": tuple(f"yolo11{s}.pt" for s in "nsmlx"),
    "segment": tuple(f"yolo11{s}-seg.pt" for s in "nsmlx"),
}
METRICS_FORMAT = "autolabel-metrics-1"
METRICS_MAX_BYTES = 1024 * 1024


class NoPolygons(Exception):
    """segment was asked but no exported box has a polygon (the route answers 409)."""


class BadMetrics(Exception):
    """A metrics file breaks a rule of the contract (the backend answers 400 bad_metrics)."""


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_train_options(task: Any, epochs: Any = EPOCHS, imgsz: Any = IMGSZ,
                        base_weights: Any = None) -> str:
    """
    Checks the #23 options; returns the base weights to use (None -> BASE_WEIGHTS[task]).
    Raises ValueError with a readable message (the route answers 400 bad_request).
    """
    if task not in TRAIN_TASKS:
        raise ValueError(f"task must be one of {list(TRAIN_TASKS)}")
    lo, hi = EPOCHS_RANGE
    if not _is_int(epochs) or not lo <= epochs <= hi:
        raise ValueError(f"epochs must be an int {lo}-{hi}")
    lo, hi = IMGSZ_RANGE
    if not _is_int(imgsz) or not lo <= imgsz <= hi or imgsz % 32:
        raise ValueError(f"imgsz must be an int {lo}-{hi} and a multiple of 32")
    if base_weights is None:
        return BASE_WEIGHTS[task]
    if base_weights not in ALLOWED_WEIGHTS[task]:
        other = next(t for t in TRAIN_TASKS if t != task)
        if base_weights in ALLOWED_WEIGHTS[other]:
            raise ValueError(f"base_weights {base_weights} is for {other}, not {task}")
        raise ValueError(f"base_weights must be one of {list(ALLOWED_WEIGHTS[task])} for {task}")
    return base_weights


# ------------------------------------------------------------------ metrics file

def _bad(msg: str) -> BadMetrics:
    return BadMetrics(f"metrics file: {msg}")


def _check_maps(value: Any, key: str) -> dict:
    if not isinstance(value, dict):
        raise _bad(f"metrics.{key} must be an object with map50 and map50_95")
    out = {}
    for k in ("map50", "map50_95"):
        v = value.get(k)
        if (not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
                or not 0 <= v <= 1):
            raise _bad(f"metrics.{key}.{k} must be a number in [0, 1]")
        out[k] = float(v)
    return out


def _check_utc(value: Any) -> None:
    msg = "metrics.finished_at must be an ISO UTC string"
    if not isinstance(value, str):
        raise _bad(msg)
    try:
        t = datetime.fromisoformat(value)
    except ValueError:
        raise _bad(msg) from None
    if t.utcoffset() != timedelta(0):
        raise _bad(msg)


def read_metrics(data: bytes, task: str) -> dict:
    """
    Reads a `<run>_metrics.json` written by the notebook. Returns {"metrics": ..., "split": ...}
    (unknown keys inside `metrics` are dropped, `split` is returned unchanged).
    Raises BadMetrics(message) when a contract rule is broken.
    """
    if not data:
        raise _bad("file is empty")
    if len(data) > METRICS_MAX_BYTES:
        raise _bad("file is larger than 1 MB")
    try:
        doc = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise _bad("not UTF-8 JSON") from None
    if not isinstance(doc, dict):
        raise _bad("must be a JSON object")
    if doc.get("format") != METRICS_FORMAT:
        raise _bad(f"format must be {METRICS_FORMAT}")
    if doc.get("task") != task:
        raise _bad(f"task is {doc.get('task')!r}, the import's task is {task!r}")
    split = doc.get("split")
    if not isinstance(split, dict):
        raise _bad("split must be an object (the package's split.json)")
    m = doc.get("metrics")
    if not isinstance(m, dict):
        raise _bad("metrics must be an object")

    box = _check_maps(m.get("box"), "box")
    if task == "segment":
        mask = _check_maps(m.get("mask"), "mask")
    else:
        if m.get("mask") is not None:
            raise _bad("metrics.mask must be null for detect")
        mask = None
    for k in ("epochs", "imgsz"):
        if not _is_int(m.get(k)) or m[k] < 1:
            raise _bad(f"metrics.{k} must be a positive int")
    if not isinstance(m.get("base_weights"), str):
        raise _bad("metrics.base_weights must be a string")
    _check_utc(m.get("finished_at"))
    metrics = {"box": box, "mask": mask, "epochs": m["epochs"], "imgsz": m["imgsz"],
               "base_weights": m["base_weights"], "finished_at": m["finished_at"]}
    return {"metrics": metrics, "split": split}


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


def notebook(project_name: str, task: str, split: dict, epochs: int = EPOCHS,
             imgsz: int = IMGSZ, base_weights: str | None = None) -> dict:
    """The 4-cell Colab notebook of #23 as a JSON-ready dict."""
    weights = base_weights or BASE_WEIGHTS[task]
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
        f"3. Run the **check + download** cell; it checks the model on the val images and saves "
        f"`{run}_best.pt` and `{run}_metrics.json`. Import both in the app.\n"
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
        f'model = YOLO("{weights}")\n'
        f'model.train(data="/content/dataset/data.yaml", imgsz={imgsz}, epochs={epochs}, '
        f'project="/content/runs", name="{run}", exist_ok=True)\n'
    )
    mask = ('{"map50": float(m.seg.map50), "map50_95": float(m.seg.map)}'
            if task == "segment" else "None")
    download = (
        "import datetime, json, shutil\n"
        "from google.colab import files\n"
        "from ultralytics import YOLO\n"
        f'shutil.copy("/content/runs/{run}/weights/best.pt", "/content/{run}_best.pt")\n'
        f'm = YOLO("/content/{run}_best.pt").val(data="/content/dataset/data.yaml", '
        f'imgsz={imgsz}, split="val")\n'
        'with open("/content/dataset/split.json", encoding="utf-8") as f:\n'
        "    split = json.load(f)\n"
        "metrics = {\n"
        '    "box": {"map50": float(m.box.map50), "map50_95": float(m.box.map)},\n'
        f'    "mask": {mask},\n'
        f'    "epochs": {epochs}, "imgsz": {imgsz}, "base_weights": "{weights}",\n'
        '    "finished_at": datetime.datetime.now(datetime.timezone.utc)'
        '.strftime("%Y-%m-%dT%H:%M:%SZ"),\n'
        "}\n"
        f'with open("/content/{run}_metrics.json", "w", encoding="utf-8") as f:\n'
        f'    json.dump({{"format": "{METRICS_FORMAT}", "source": "colab", "task": "{task}", '
        '"split": split, "metrics": metrics}, f, indent=1)\n'
        'print("val box mAP50", metrics["box"]["map50"], "mask", metrics["mask"])\n'
        f'files.download("/content/{run}_best.pt")\n'
        f'files.download("/content/{run}_metrics.json")\n'
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
                        image_bytes: Callable[[dict], bytes], *, epochs: int = EPOCHS,
                        imgsz: int = IMGSZ, base_weights: str | None = None) -> bytes:
    """
    API JSON shapes in (as build_zip); returns the #23 zip. Raises ValueError (bad task or
    option), NothingToExport (no done/empty image), NoPolygons. Calls image_bytes only for
    exported images, and only after every check passed. base_weights None = BASE_WEIGHTS[task].
    """
    base_weights = check_train_options(task, epochs, imgsz, base_weights)
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
        zf.writestr("train.ipynb", json.dumps(notebook(project_name, task, split, epochs, imgsz,
                                                         base_weights), indent=1))
        for im in chosen:
            part, name = part_of[im["id"]], names[im["id"]]
            zf.writestr(f"labels/{part}/{PurePosixPath(name).stem}.txt", labels[im["id"]])
            zf.writestr(f"images/{part}/{name}", image_bytes(im))
    return buf.getvalue()
