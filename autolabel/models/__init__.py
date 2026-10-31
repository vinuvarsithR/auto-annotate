"""
autolabel.models - the detector registry (docs/CONTRACT.md section 3).

list_detectors() -> names the backend may put in POST .../jobs "model".
get_detector(name) -> the detector object. Each one is built once and reused,
so real models (added later) load their weights only the first time.

Only "fake" is registered for now. Real detectors will import torch inside
their own constructors, never at module level, so importing this package
stays cheap and works without a GPU.
"""

from __future__ import annotations

from typing import Callable

from autolabel.models.base import ClassSpec, Detector, Prediction


def _make_fake(**options) -> Detector:
    from autolabel.models.fake import FakeDetector
    return FakeDetector(**options)


# name -> factory. Factories import their module lazily.
_FACTORIES: dict[str, Callable[..., Detector]] = {
    "fake": _make_fake,
}

_CACHE: dict[str, Detector] = {}


def list_detectors() -> list[str]:
    return list(_FACTORIES)


def get_detector(name: str, **options) -> Detector:
    """Return the detector called `name`, building it on first use.

    Options are only used the first time a name is requested.
    """
    if name not in _FACTORIES:
        raise ValueError(f"unknown detector {name!r}; choose one of {list_detectors()}")
    if name not in _CACHE:
        _CACHE[name] = _FACTORIES[name](**options)
    return _CACHE[name]


__all__ = ["ClassSpec", "Detector", "Prediction", "get_detector", "list_detectors"]
