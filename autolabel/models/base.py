"""
base.py - the shared types every detector uses (docs/CONTRACT.md section 3).

Plain-English summary
---------------------
* ClassSpec: one class the user wants found (short name + text description).
* Prediction: one box a model drew. It IS a Box (from evaluate.py), so the
  existing scoring, merge and threshold code works on it unchanged.
* Example: one reviewed box a detector learns from (examples mode, CONTRACT #9).
* ExampleRun: one job's prototypes; built at job start, kept in memory, never stored.
* Detector: anything with a `name` and a `detect(image, image_id, classes)` method.
  Detectors with mode "examples" also have `prepare_examples(classes, examples)`.

Other sessions import these types; nobody redefines them.
No torch / GPU code here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    source: str = ""               # "owlv2", "owlv2_image", "yoloe_examples", "gdino", "pipeline", ...
    # mask and polygon are left out of hash/==, because a dict or list is not hashable
    # and a frozen dataclass hashes every field (CONTRACT.md section 3).
    mask: dict | None = field(default=None, hash=False, compare=False)     # COCO RLE or None
    polygon: list | None = field(default=None, hash=False, compare=False)  # [[x, y], ...] 0..1, or None


@dataclass(frozen=True)
class Example:
    image_id: str
    path: str                      # image on disk
    category: str                  # class name
    bbox: tuple                    # (x, y, w, h) px, clipped to the image, area > 0


class ExampleRun(Protocol):        # one job's prototypes; built at job start, kept in memory
    classes: list[ClassSpec]       # the classes that will run, in input order
    used: dict[str, int]           # class name -> examples used (classes that run only)
    unusable: list[str]            # class names the detector rejected all examples for
    skip_reasons: dict[str, str]   # optional: class in `unusable` -> reason for #9 `result`

    def detect(self, image: Image.Image, image_id: str) -> list[Prediction]: ...


class Detector(Protocol):
    name: str                      # the string used in POST .../jobs "model"

    def detect(self, image: Image.Image, image_id: str,
               classes: list[ClassSpec]) -> list[Prediction]: ...

    # only detectors with mode "examples" have this one:
    def prepare_examples(self, classes: list[ClassSpec],
                         examples: dict[str, list[Example]]) -> ExampleRun: ...
