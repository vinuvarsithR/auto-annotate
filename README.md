# Auto-Annotate

A locally hosted, multi-user app that auto-labels images for object detection.
Upload images, describe your classes in plain text, get suggested boxes, review
them in the browser, export to COCO / YOLO, and sync to Roboflow.

**Status: Phase 1 (OWLv2 script + scoring).**

---

## Phase 1: run OWLv2 and score it

### 1. Get a provisional gold set from Roboflow

In Roboflow, generate a dataset version of the project with **no augmentation
and no resize**. Then:

```bash
python scripts/fetch_gold_from_roboflow.py \
    --workspace <workspace-slug> --project <project-slug> --version <N> \
    --name eot_crane_provisional --n 50
```

This picks 50 images (rarest classes first), copies them to
`gold_sets/eot_crane_provisional/`, and writes `annotations.json`.
The labels are copied as-is, not re-checked, so treat the scores as approximate.

### 2. Add `classes.yaml`

Each class needs a plain-English description; that's what OWLv2 searches for.
Use `aliases` to map other spellings in Roboflow to the canonical name.

### 3. Run

```bash
python scripts/run_owlv2.py --gold gold_sets/eot_crane_provisional
```

The first run downloads the model (~620 MB). Results go to
`runs/eot_crane_provisional/`: `predictions.json` (every box scoring 0.05 or
higher) and `sweep.csv` (precision and recall per class at each threshold).

To try new descriptions, edit `classes.yaml` and run again. To re-score saved
predictions without re-running the model (e.g. with a different `--iou`), add
`--score-only`.

### Reading the output

| Column | Meaning |
|---|---|
| precision | Of the boxes drawn, how many were right. Low = lots of junk to delete. |
| recall | Of the real objects, how many were found. Low = lots of boxes to draw by hand. |
| recall (class ignored) | Found the object, even if the class was wrong. Much higher than recall = the model sees the object but confuses classes. |
| boxes drawn | How many boxes a reviewer would see at that threshold |

Pick the threshold with high recall and precision you can live with.

### Tests

```bash
pytest
```

---

## Hardware

| Machine | GPU | VRAM | Architecture | OS |
|---|---|---|---|---|
| A | RTX 3060 | 12 GB | Ampere (`sm_86`) | Linux |
| B | RTX 5060 Ti | 16 GB | Blackwell (`sm_120`) | Windows |

Both cards need **PyTorch 2.7 or newer built with CUDA 12.8 or newer** (`cu128`).
The 5060 Ti will not work with older builds: you'll see
`no kernel image is available for execution on the device`.
The 3060 works with older builds too, but use the same version on both machines
so results match.

---

## Setup (do this on each machine)

### 1. Install Python 3.11

- **Linux:** `sudo apt install python3.11 python3.11-venv` (or your distro's equivalent)
- **Windows:** install from python.org and tick "Add python.exe to PATH"

### 2. Update the NVIDIA driver

CUDA 12.8 needs a recent driver. Version 570 or newer *should* be enough
(**unverified guess**; the GPU check prints your actual version).

- **Linux:** `nvidia-smi` shows your driver version in the top row
- **Windows:** install the latest Game Ready or Studio driver from nvidia.com

### 3. Create a virtual environment

A virtual environment is a private folder of Python packages for this project,
so it can't break anything else on the machine.

```bash
# Linux
python3.11 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 4. Install PyTorch (same command on both machines)

PyTorch is installed separately because the normal `pip install torch` may pick
a build without GPU support for Blackwell.

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

### 5. Install the rest

```bash
pip install -r requirements.txt
```

### 6. Run the GPU check

```bash
python scripts/check_gpu.py
```

What a good result looks like:

| Line | Expected on the 3060 | Expected on the 5060 Ti |
|---|---|---|
| `built with CUDA` | 12.8 or newer | 12.8 or newer |
| `architecture` | `sm_86` | `sm_120` |
| `kernels in this build` | yes | yes |
| `matmul test` | PASS | PASS |

If `kernels in this build` says **NO**, your PyTorch is too old: repeat step 4.
If no GPU passes, the script says so and the app will fall back to CPU (slow).

### 7. Set up secrets

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and paste your Roboflow API key. `.env` is in `.gitignore`, so it
never gets committed. Never paste the key into code, chat, or a commit.

---

## Repo layout

```
auto-annotate/
├── README.md            this file
├── .gitignore           keeps secrets, data and model weights out of git
├── .env.example         template for .env (no real values)
├── requirements.txt     Python packages (PyTorch is installed separately)
├── scripts/
│   ├── check_gpu.py                 Phase 0 hardware check
│   ├── fetch_gold_from_roboflow.py  Phase 1: provisional gold set from Roboflow
│   └── run_owlv2.py                 Phase 1: run OWLv2 + score + threshold sweep
├── autolabel/           shared library (reused by the app later)
│   ├── owlv2.py         loads and runs OWLv2
│   ├── gold.py          loads a gold set, maps class aliases
│   ├── coco.py          merges Roboflow splits, samples gold images
│   └── evaluate.py      IoU, matching, precision/recall, sweep
├── app/                 backend + UI (Phase 2 onward)
├── tests/               tests for the scoring and COCO code
├── runs/                Phase 1 outputs (gitignored)
├── models/              downloaded model weights (gitignored)
└── gold_sets/           hand-checked images used to score the auto-labeler
    ├── README.md        project table, class names, labeling rules
    ├── _template/       copy this to start a new gold set
    │   ├── classes.yaml     class names + text descriptions
    │   ├── annotations.json hand labels in COCO format
    │   └── images/          the images
    └── <name>/          one folder per gold set, same layout
```

---

## Gold sets

A gold set is 50–100 images with carefully checked labels. It's the answer key:
every phase from Phase 1 onward scores the auto-labels against it, so mistakes
here make every later number wrong.

There is one gold set per combination of classes and camera view, so several
Roboflow projects can share one. Our 14 projects fall into two class groups
(blade root and EOT crane), split further by camera view.
See [`gold_sets/README.md`](gold_sets/README.md) for the project table,
canonical class names and labeling rules.

---

## Roadmap

| Phase | What gets built | Status |
|---|---|---|
| 0 | Setup, GPU check, repo structure, gold-set folders | Done (Linux) |
| 1 | Standalone OWLv2 script + precision/recall vs. gold set + threshold sweep | **This phase** |
| 2 | FastAPI backend: upload, class descriptions, job queue, results store | |
| 3 | Review UI, COCO/YOLO export, Roboflow upload | |
| 4 | SAM masks, second model ensemble, VLM verifier, confidence routing | |
| 5 | Train YOLO / RF-DETR on auto-labels, flag likely label errors | |
| 6 | Docker, login, docs, demo script | |
