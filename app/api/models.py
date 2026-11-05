"""
models.py - endpoint 16: the detector names and modes that POST .../jobs accepts.

`models` is exactly autolabel.models.list_detectors() (via app.state.list_detectors, the same
function endpoint 9 uses to check the "model" field). `modes` has one key per model, default
mode first, [] when the detector has no modes.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/models")
def list_models(request: Request):
    models = list(request.app.state.list_detectors())
    return {"models": models,
            "modes": {name: list(request.app.state.detector_modes(name)) for name in models}}
