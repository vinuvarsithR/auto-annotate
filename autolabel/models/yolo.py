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
* Runs on CPU by default (`device` option). get_detector() keeps one loaded model
  per process. ultralytics / torch are imported in the constructor, not at module level.
"""

from __future__ import annotations

import os

from PIL import Image

from autolabel.models.base import ClassSpec, Prediction

ENV_WEIGHTS = "AUTOLABEL_YOLO_WEIGHTS"
DEFAULT_MIN_SCORE = 0.05


def _as_list(x) -> list:
    """torch tensor / numpy array / list -> plain Python list."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    return x.tolist() if hasattr(x, "tolist") else list(x)


class YoloDetector:
    name = "yolo"

    def __init__(self, weights: str | None = None, device: str = "cpu",
                 min_score: float = DEFAULT_MIN_SCORE, imgsz: int | None = None,
                 model=None) -> None:
        """model: anything with `.names` ({index: name}) and `.predict(...)` like
        ultralytics.YOLO. Left as None, the weights file is loaded with ultralytics."""
        self.device = device
        self.min_score = min_score
        self.imgsz = imgsz
        if model is None:
            weights = weights or os.environ.get(ENV_WEIGHTS, "").strip()
            if not weights:
                raise ValueError(f"yolo detector needs weights: set {ENV_WEIGHTS}=path/to/best.pt")
            # A bare name like "yolo11n.pt" may be downloaded by ultralytics; a path must exist.
            if not os.path.isfile(weights) and ("/" in weights or "\\" in weights):
                raise ValueError(f"yolo weights not found: {weights}")
            from ultralytics import YOLO
            model = YOLO(weights)
        self.model = model
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
                      device=self.device, verbose=False)
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        results = self.model.predict(image, **kwargs)
        out: list[Prediction] = []
        for r in results:
            b = r.boxes
            for (x1, y1, x2, y2), score, cls in zip(_as_list(b.xyxy), _as_list(b.conf),
                                                   _as_list(b.cls)):
                name = wanted.get(int(cls))
                if name is None:
                    continue
                score = min(1.0, max(0.0, float(score)))
                if score < floors[name]:
                    continue
                out.append(Prediction(
                    image_id=image_id,
                    category=name,
                    bbox=(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                    score=score,
                    source=self.name,
                    mask=None,
                ))
        return out
