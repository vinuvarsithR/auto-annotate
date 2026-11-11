"""
yolo.py - a trained YOLO (Ultralytics) detector, behind the same interface as "fake" / "owlv2".

Plain-English summary
---------------------
* Weights: the `weights` option, else the environment variable AUTOLABEL_YOLO_WEIGHTS
  (e.g. a best.pt trained on Colab with classes person, rb, sb, biax).
* Classes are matched BY NAME, never by index: a model class is used only when a
  project class has exactly the same name. Model classes the project does not have
  are ignored; project classes the model does not know get no boxes.
  (ClassSpec.description is not used: YOLO has no text prompts.)
* Score floor per class: the class's `min_score`, else DEFAULT_MIN_SCORE (0.05,
  recall-first, same default as "owlv2"; not tuned for YOLO).
* Image size: the `imgsz` option, else env AUTOLABEL_YOLO_IMGSZ, else the size the
  weights were trained with (checkpoint train_args), else 640. YOLO accuracy drops when
  run at a different size than it was trained at.
* Runs on CPU by default (`device` option). get_detector() keeps one loaded model
  per process. ultralytics / torch are imported in the constructor, not at module level.
"""

from __future__ import annotations

import os

from PIL import Image

from autolabel.models.base import ClassSpec, Prediction

ENV_WEIGHTS = "AUTOLABEL_YOLO_WEIGHTS"
ENV_IMGSZ = "AUTOLABEL_YOLO_IMGSZ"
DEFAULT_MIN_SCORE = 0.05
DEFAULT_IMGSZ = 640


def trained_imgsz(model):
    """The training image size stored in an ultralytics model, or None."""
    ckpt = getattr(model, "ckpt", None)
    args = ckpt.get("train_args") if isinstance(ckpt, dict) else None
    size = (args or {}).get("imgsz")
    if size is None:
        size = (getattr(model, "overrides", None) or {}).get("imgsz")
    return size


def choose_imgsz(option, model):
    """option > AUTOLABEL_YOLO_IMGSZ > checkpoint train_args > 640."""
    if option:
        return option
    env = os.environ.get(ENV_IMGSZ, "").strip()
    if env:
        try:
            value = int(env)
        except ValueError:
            raise ValueError(f"{ENV_IMGSZ} must be a whole number, got {env!r}") from None
        if value <= 0:
            raise ValueError(f"{ENV_IMGSZ} must be positive, got {value}")
        return value
    return trained_imgsz(model) or DEFAULT_IMGSZ


def _as_list(x) -> list:
    """torch tensor / numpy array / list -> plain Python list."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    return x.tolist() if hasattr(x, "tolist") else list(x)


def disable_autoinstall() -> None:
    """Never pip-install during a job.

    ultralytics patches PIL.Image.open: after that, any file Pillow cannot identify runs
    `pip install pi-heif` in the live environment (VERIFIED 2026-11-05, 8.4.199). The env variable
    is read once at import, so also set the flag, in case ultralytics was imported earlier.
    """
    os.environ["YOLO_AUTOINSTALL"] = "false"
    from ultralytics.utils import checks

    checks.AUTOINSTALL = False


class YoloDetector:
    name = "yolo"
    source = "yolo"                 # Prediction.source; the trained detector overrides it
    predict_options: dict = {}      # extra predict() keyword arguments (trained segment: retina_masks)

    def __init__(self, weights: str | None = None, device: str = "cpu",
                 min_score: float = DEFAULT_MIN_SCORE, imgsz: int | None = None,
                 model=None) -> None:
        """model: anything with `.names` ({index: name}) and `.predict(...)` like
        ultralytics.YOLO. Left as None, the weights file is loaded with ultralytics."""
        self.device = device
        self.min_score = min_score
        if model is None:
            weights = weights or os.environ.get(ENV_WEIGHTS, "").strip()
            if not weights:
                raise ValueError(f"yolo detector needs weights: set {ENV_WEIGHTS}=path/to/best.pt")
            # A bare name like "yolo11n.pt" may be downloaded by ultralytics; a path must exist.
            if not os.path.isfile(weights) and ("/" in weights or "\\" in weights):
                raise ValueError(f"yolo weights not found: {weights}")
            disable_autoinstall()
            from ultralytics import YOLO
            model = YOLO(weights)
        self.model = model
        self.imgsz = choose_imgsz(imgsz, model)
        names = model.names
        pairs = names.items() if isinstance(names, dict) else enumerate(names)
        # model class name -> model class index
        self.index_by_name: dict[str, int] = {str(n): int(i) for i, n in pairs}

    def detect(self, image: Image.Image, image_id: str,
               classes: list[ClassSpec]) -> list[Prediction]:
        floors = {c.name: (self.min_score if c.min_score is None else c.min_score)
                  for c in classes if c.name in self.index_by_name}
        if not floors:
            return []
        wanted = {self.index_by_name[n]: n for n in floors}   # model index -> project class name
        kwargs = dict(conf=min(floors.values()), classes=sorted(wanted),
                      device=self.device, verbose=False, imgsz=self.imgsz, **self.predict_options)
        results = self.model.predict(image, **kwargs)
        out: list[Prediction] = []
        for r in results:
            b = r.boxes
            for i, ((x1, y1, x2, y2), score, cls) in enumerate(zip(_as_list(b.xyxy), _as_list(b.conf),
                                                                   _as_list(b.cls))):
                name = wanted.get(int(cls))
                if name is None:
                    continue
                score = min(1.0, max(0.0, float(score)))
                if score < floors[name]:
                    continue
                bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                bbox, polygon = self._shape(r, i, bbox, image)
                out.append(Prediction(
                    image_id=image_id,
                    category=name,
                    bbox=bbox,
                    score=score,
                    source=self.source,
                    mask=None,
                    polygon=polygon,
                ))
        return out

    def _shape(self, result, index: int, bbox: tuple, image: Image.Image):
        """(bbox, polygon) for box `index` of `result`. Plain YOLO: the model box, no polygon."""
        return bbox, None
