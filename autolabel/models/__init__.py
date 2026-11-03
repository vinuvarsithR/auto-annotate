"""
autolabel.models - the detector registry (docs/CONTRACT.md section 3).

list_detectors() -> names the backend may put in POST .../jobs "model".
get_detector(name) -> the detector object. Each one is built once and reused,
so real models (added later) load their weights only the first time.

"fake" is always available. Real detectors are opt-in through the environment
variable AUTOLABEL_DETECTORS (comma-separated), e.g. AUTOLABEL_DETECTORS=owlv2,yolo
("yolo" also needs AUTOLABEL_YOLO_WEIGHTS=path/to/best.pt).
Unset -> only "fake", so tests stay fast. Real detectors import torch inside
their own constructors, never at module level, so importing this package
stays cheap and works without a GPU.
"""

from __future__ import annotations

import os
from typing import Callable

from autolabel.models.base import ClassSpec, Detector, Prediction


def _make_fake(**options) -> Detector:
    from autolabel.models.fake import FakeDetector
    return FakeDetector(**options)


def _make_owlv2(**options) -> Detector:
    from autolabel.models.owlv2 import Owlv2TextDetector
    return Owlv2TextDetector(**options)


def _make_yolo(**options) -> Detector:
    from autolabel.models.yolo import YoloDetector
    return YoloDetector(**options)


# name -> factory. Factories import their module lazily.
_FACTORIES: dict[str, Callable[..., Detector]] = {
    "fake": _make_fake,
    "owlv2": _make_owlv2,
    "yolo": _make_yolo,
}
ALWAYS_ON = ("fake",)
ENV_VAR = "AUTOLABEL_DETECTORS"

_CACHE: dict[str, Detector] = {}


def list_detectors() -> list[str]:
    """"fake" plus the real detectors named in AUTOLABEL_DETECTORS (read on every call)."""
    extra = [n.strip() for n in os.environ.get(ENV_VAR, "").split(",") if n.strip()]
    return [n for n in _FACTORIES if n in ALWAYS_ON or n in extra]


def get_detector(name: str, **options) -> Detector:
    """Return the detector called `name`, building it on first use.

    Options are only used the first time a name is requested.
    """
    if name not in list_detectors():
        hint = f" (set {ENV_VAR}={name} to enable it)" if name in _FACTORIES else ""
        raise ValueError(f"unknown detector {name!r}; choose one of {list_detectors()}{hint}")
    if name not in _CACHE:
        _CACHE[name] = _FACTORIES[name](**options)
    return _CACHE[name]


__all__ = ["ClassSpec", "Detector", "Prediction", "get_detector", "list_detectors"]
