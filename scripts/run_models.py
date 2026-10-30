"""
run_models.py - run one detector on the VAL and TEST images and save every box
above a low score (needed for mAP and for picking thresholds on val).

    python scripts/run_models.py owl_text
    python scripts/run_models.py owl_image     # step B: image-guided rb/sb/biax, text person
    python scripts/run_models.py gdino
    python scripts/run_models.py yolo

Writes runs/compare/<model>/val.json, test.json (lists of {file_name, category,
bbox [x,y,w,h], score}) and meta.json (speed, peak VRAM, settings).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from autolabel.gold import load_classes  # noqa: E402

TEST_DIR = ROOT / "gold_sets/eot_crane_test50"
SPLIT_DIR = ROOT / "data/splits/eot_crane_v1"
IMAGE_CLASSES = ROOT / "experiments/image_queries/B1.yaml"
YOLO_WEIGHTS = ROOT / "runs/yolo/eot_crane_v1_yolo26s/weights/best.pt"


def image_lists() -> dict[str, list[Path]]:
    info = json.loads((SPLIT_DIR / "split_info.json").read_text())
    src = {p.name: p for p in (ROOT / info["export"]).rglob("*.jpg")}
    val = [src[im["file_name"]] for im in json.loads((SPLIT_DIR / "val.json").read_text())["images"]]
    test = [TEST_DIR / "images" / im["file_name"] for im in json.loads((TEST_DIR / "annotations.json").read_text())["images"]]
    return {"val": sorted(val), "test": sorted(test)}


def make_runner(model: str, save_threshold: float):
    """Returns (detect(image) -> list of (category, bbox, score), peak_vram(), settings dict)."""
    if model == "owl_text":
        from autolabel.owlv2 import DEFAULT_MODEL, Owlv2Detector

        classes = load_classes(TEST_DIR / "classes.yaml")
        det = Owlv2Detector(DEFAULT_MODEL)
        prompts = [c.description for c in classes]

        def run(im):
            return [(classes[d.label].name, d.bbox, d.score) for d in det.detect(im, prompts, save_threshold)]

        return run, det.peak_vram_gb, {"model_id": DEFAULT_MODEL, "prompts": dict(zip([c.name for c in classes], prompts))}

    if model == "owl_image":
        from autolabel.owlv2 import DEFAULT_MODEL, Owlv2Detector

        classes = load_classes(IMAGE_CLASSES)
        det = Owlv2Detector(DEFAULT_MODEL)
        text_cls = [c for c in classes if not c.examples]
        img_cls = [c for c in classes if c.examples]
        queries, overlaps = [], {}
        for c in img_cls:
            qs = []
            for e in c.examples:
                with Image.open(e.image) as pil:
                    q, ov = det.query_embedding(pil.convert("RGB"), e.bbox)
                qs.append(q)
                overlaps.setdefault(c.name, []).append(round(ov, 3))
            queries.append(qs)
        print("Example box vs model's best-matching box (IoU):", overlaps)

        def run(im):
            out = []
            if text_cls:
                for d in det.detect(im, [c.description for c in text_cls], save_threshold):
                    out.append((text_cls[d.label].name, d.bbox, d.score))
            for d in det.detect_image_guided(im, queries, save_threshold):
                out.append((img_cls[d.label].name, d.bbox, d.score))
            return out

        settings = {"model_id": DEFAULT_MODEL, "text_classes": [c.name for c in text_cls],
                    "image_classes": {c.name: [str(e.image.name) for e in c.examples] for c in img_cls},
                    "example_overlap_iou": overlaps}
        return run, det.peak_vram_gb, settings

    if model == "gdino":
        from autolabel.gdino import DEFAULT_MODEL, GroundingDinoDetector

        classes = load_classes(TEST_DIR / "classes.yaml")
        det = GroundingDinoDetector(DEFAULT_MODEL)
        descs = [c.description for c in classes]

        def run(im):
            return [(classes[d.label].name, d.bbox, d.score) for d in det.detect(im, descs, save_threshold)]

        return run, det.peak_vram_gb, {"model_id": DEFAULT_MODEL, "prompt": __import__("autolabel.gdino", fromlist=["x"]).build_prompt(descs)[0]}

    if model == "yolo":
        import torch
        from ultralytics import YOLO

        yolo = YOLO(str(YOLO_WEIGHTS))

        def run(im):
            r = yolo.predict(im, imgsz=1280, conf=0.001, verbose=False)[0]
            out = []
            for (x1, y1, x2, y2), s, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
                out.append((yolo.names[int(c)], (x1, y1, x2 - x1, y2 - y1), s))
            return out

        def vram():
            return torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else None

        return run, vram, {"weights": str(YOLO_WEIGHTS.relative_to(ROOT)), "imgsz": 1280, "conf": 0.001,
                           "licence": "Ultralytics AGPL-3.0"}

    raise SystemExit(f"unknown model {model}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=["owl_text", "owl_image", "gdino", "yolo"])
    ap.add_argument("--save-threshold", type=float, default=0.01)
    args = ap.parse_args()

    out_dir = ROOT / "runs/compare" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    run, vram, settings = make_runner(args.model, args.save_threshold)

    meta = {"settings": settings, "save_threshold": args.save_threshold}
    for split, paths in image_lists().items():
        preds, start = [], time.time()
        for path in paths:
            with Image.open(path) as pil:
                im = pil.convert("RGB")
            for cat, bbox, score in run(im):
                preds.append({"file_name": path.name, "category": cat,
                              "bbox": [round(v, 2) for v in bbox], "score": round(score, 5)})
        secs = time.time() - start
        (out_dir / f"{split}.json").write_text(json.dumps(preds))
        meta[f"{split}_images"] = len(paths)
        meta[f"{split}_img_per_s"] = round(len(paths) / secs, 2)
        print(f"{args.model} {split}: {len(paths)} images, {len(preds)} boxes, {len(paths) / secs:.2f} img/s")
    meta["peak_vram_gb"] = round(vram(), 2) if vram() is not None else None
    print(f"Peak GPU memory (this process): {meta['peak_vram_gb']} GB")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
