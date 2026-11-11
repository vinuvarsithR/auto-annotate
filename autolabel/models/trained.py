"""
trained.py - weights the user trained on Colab and uploaded (CONTRACT #24 - #26, section 3).

Plain-English summary
---------------------
* inspect_weights(path) opens a `.pt` with ultralytics and reports its task ("detect",
  "segment", ...), its class names in index order and its training image size. It never
  downloads: a missing file is an error. Any load failure is a one-line ValueError.
  It does not judge the task; the backend decides what is allowed.
* register_trained(name, path, task) makes `name` usable with get_detector(name). It is not
  added to list_detectors() (the backend lists a project's own models itself) and has no modes.
  The model is loaded the first time get_detector(name) asks for it, then kept.
  Registering the same name + path again changes nothing; a new path drops the loaded model.
* The trained detector follows the `yolo` rules (classes by name, score floors with 0.05 default,
  CPU, autoinstall off), uses the checkpoint's own image size and sets source "yolo_trained".
  Task "segment": masks become polygons with yoloe.mask_to_polygon (one part, or the largest
  if it holds >= 90%); with a polygon the bbox is the polygon's, without one the model's box.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

from PIL import Image

from autolabel.models.yolo import DEFAULT_IMGSZ, DEFAULT_MIN_SCORE, YoloDetector, disable_autoinstall, \
    trained_imgsz

SOURCE = "yolo_trained"


@dataclass(frozen=True)
class WeightsInfo:
    task: str                      # the checkpoint's ultralytics task: "detect", "segment", "pose", ...
    class_names: list[str]         # model names in index order
    imgsz: int | None              # train_args imgsz (as yolo.trained_imgsz), else None


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _load(path: str):
    """ultralytics.YOLO(path) on an existing file; failures -> one-line ValueError."""
    if not os.path.isfile(path):
        # Checked first: given a bare name like "yolo11n.pt", ultralytics would download it.
        raise ValueError(_one_line(f"weights file not found: {path}"))
    try:
        disable_autoinstall()
        from ultralytics import YOLO
        return YOLO(path)
    except Exception as e:  # torch / pickle / ultralytics errors vary widely
        reason = _one_line(str(e)) or type(e).__name__
        raise ValueError(_one_line(f"cannot load weights {os.path.basename(path)}: {reason}"[:300])) from None


def _class_names(names) -> list[str]:
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def inspect_weights(path: str, model=None) -> WeightsInfo:
    """Task, class names (index order) and training image size of a checkpoint."""
    if model is None:
        model = _load(path)
    try:
        size = trained_imgsz(model)
        return WeightsInfo(task=str(model.task), class_names=_class_names(model.names),
                           imgsz=int(size) if isinstance(size, (int, float)) else None)
    except Exception as e:
        raise ValueError(_one_line(f"cannot read weights {os.path.basename(path)}: {e}")) from None


class TrainedDetector(YoloDetector):
    """A registered trained model: `yolo` rules + segment polygons, source "yolo_trained"."""
    source = SOURCE

    def __init__(self, name: str, model, task: str, device: str = "cpu",
                 min_score: float = DEFAULT_MIN_SCORE) -> None:
        super().__init__(device=device, min_score=min_score,
                         imgsz=trained_imgsz(model) or DEFAULT_IMGSZ, model=model)
        self.name = name
        self.task = task
        self.predict_options = {"retina_masks": True} if task == "segment" else {}

    def _shape(self, result, index: int, bbox: tuple, image: Image.Image):
        if self.task != "segment" or getattr(result, "masks", None) is None:
            return bbox, None
        from autolabel.models.yoloe import mask_to_polygon, polygon_bbox

        mask = result.masks.data[index]
        if hasattr(mask, "cpu"):
            mask = mask.cpu().numpy()
        polygon = mask_to_polygon(mask)
        if polygon is None:
            return bbox, None
        return polygon_bbox(polygon, *image.size), polygon


@dataclass
class _Entry:
    path: str
    task: str
    model: object = None           # stand-in for tests; None = load `path`
    detector: TrainedDetector | None = None


_REGISTRY: dict[str, _Entry] = {}
_LOCK = threading.Lock()


def register_trained(name: str, path: str, task: str, model=None) -> None:
    """Make get_detector(name) return this model. Same name + path again: no-op."""
    with _LOCK:
        old = _REGISTRY.get(name)
        if old is not None and old.path == path:
            return
        _REGISTRY[name] = _Entry(path=path, task=task, model=model)


def unregister_trained(name: str) -> None:
    """Forget `name` and its loaded model; unknown name: no-op."""
    with _LOCK:
        _REGISTRY.pop(name, None)


def is_registered(name: str) -> bool:
    return name in _REGISTRY


def get_trained(name: str, **options) -> TrainedDetector:
    """The detector for a registered name, built on first use (options used only then)."""
    with _LOCK:
        entry = _REGISTRY.get(name)
        if entry is None:
            raise ValueError(f"unknown detector {name!r}")
        if entry.detector is None:
            model = entry.model if entry.model is not None else _load(entry.path)
            entry.detector = TrainedDetector(name, model, entry.task, **options)
        return entry.detector
