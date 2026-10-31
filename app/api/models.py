"""
models.py - endpoint 16: the detector names that POST .../jobs accepts.

The list is exactly autolabel.models.list_detectors() (via app.state.list_detectors,
the same function endpoint 9 uses to check the "model" field).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/models")
def list_models(request: Request):
    return {"models": list(request.app.state.list_detectors())}
