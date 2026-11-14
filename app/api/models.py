"""
models.py - endpoint 16: the detector names and modes that POST .../jobs accepts.

`models` is autolabel.models.list_detectors() (via app.state.list_detectors, the same
function endpoint 9 uses to check the "model" field). `modes` has one key per model, default
mode first, [] when the detector has no modes.
With ?project_id=N (Train on Colab): that project's non-deleted trained models are added at
the end, in version order, each with modes [] (unknown project -> 404), and a key `trained`
lists them as {name, source, task, uploaded_at, metrics}.
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
            rows = conn.execute(
                "SELECT * FROM trained_models WHERE project_id=? AND deleted_at IS NULL "
                "ORDER BY version", (project_id,)).fetchall()
        finally:
            conn.close()
        trained = []
        for r in rows:
            if r["name"] not in modes:
                models.append(r["name"])
            modes[r["name"]] = []
            t = db.trained_model_dict(r)
            trained.append({k: t[k] for k in ("name", "source", "task", "uploaded_at",
                                              "metrics")})
        return {"models": models, "modes": modes, "trained": trained}
    return {"models": models, "modes": modes}
