"""
models.py - endpoint 16: the detector names and modes that POST .../jobs accepts.

`models` is autolabel.models.list_detectors() (via app.state.list_detectors, the same
function endpoint 9 uses to check the "model" field). `modes` has one key per model, default
mode first, [] when the detector has no modes.
With ?project_id=N (Train on Colab): that project's non-deleted trained models are added at
the end, in version order, each with modes [] (unknown project -> 404).
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from app.api.deps import require_project
from app.db import db

router = APIRouter()


@router.get("/models")
def list_models(request: Request, project_id: int | None = None):
    models = list(request.app.state.list_detectors())
    modes = {name: list(request.app.state.detector_modes(name)) for name in models}
    if project_id is not None:
        conn = db.connect(request.app.state.db_path)
        try:
            require_project(conn, project_id)
            trained = [r["name"] for r in conn.execute(
                "SELECT name FROM trained_models WHERE project_id=? AND deleted_at IS NULL "
                "ORDER BY version", (project_id,))]
        finally:
            conn.close()
        for name in trained:
            if name not in modes:
                models.append(name)
            modes[name] = []
    return {"models": models, "modes": modes}
