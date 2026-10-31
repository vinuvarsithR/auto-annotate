"""
base.py - the shared types every detector uses (docs/CONTRACT.md section 3).

Plain-English summary
---------------------
* ClassSpec: one class the user wants found (short name + text description).
* Prediction: one box a model drew. It IS a Box (from evaluate.py), so the
  existing scoring, merge and threshold code works on it unchanged.
* Detector: anything with a `name` and a `detect(image, image_id, classes)` method.

Other sessions import these types; nobody redefines them.
No torch / GPU code here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from autolabel.evaluate import Box


@dataclass(frozen=True)
class ClassSpec:
    name: str                      # "biax"
    description: str               # text prompt
    min_score: float | None = None


@dataclass(frozen=True)
class Prediction(Box):             # Box gives: image_id: str, category: str, bbox (x,y,w,h px), score
    source: str = ""               # "owlv2", "owlv2_image", "gdino", "pipeline", ...
    mask: dict | None = None       # COCO RLE or None


class Detector(Protocol):
    name: str                      # the string used in POST .../jobs "model"

    def detect(self, image: Image.Image, image_id: str,
               classes: list[ClassSpec]) -> list[Prediction]: ...
