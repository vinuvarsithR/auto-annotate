"""
main.py - the FastAPI app.

Run it (from the repo root, inside .venv):
    uvicorn app.main:app --port 8000

* API under /api (docs/CONTRACT.md section 2), UI from app/static at /.
* data/app.db and data/uploads/ are created on startup.
* A background worker thread runs detect, roboflow_push and import jobs (app/worker/worker.py).
* create_app(...) lets tests use a temp folder, a fake detector and no thread.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import annotations, classes, images, imports, jobs, models, projects
from app.api.errors import install_error_handlers
from app.db import db
from app.worker.worker import ImporterLoader, PipelineLoader, PusherLoader, Worker

STATIC_DIR = Path(__file__).parent / "static"


def default_list_detectors() -> list[str]:
    from autolabel.models import list_detectors
    return list(list_detectors())


def clean_imports(imports_dir: Path, conn) -> int:
    """#22: delete upload folders no queued job needs (left by a crash or restart)."""
    if not imports_dir.is_dir():
        return 0
    queued = {str(r[0]) for r in conn.execute(
        "SELECT id FROM jobs WHERE kind='import' AND status='queued'")}
    removed = 0
    for d in imports_dir.iterdir():
        if d.is_dir() and d.name not in queued:
            shutil.rmtree(d, ignore_errors=True)
            removed += 1
    return removed


def default_detector_modes(name: str) -> list[str]:
    try:
        from autolabel.models import detector_modes
    except ImportError:  # models code without modes yet: same as a detector with no modes
        return []
    return list(detector_modes(name))


def create_app(data_dir: str | Path = "data", *, start_worker: bool = True,
               load_pipeline: PipelineLoader | None = None,
               load_pusher: PusherLoader | None = None,
               load_importers: ImporterLoader | None = None,
               list_detectors: Callable[[], list[str]] | None = None,
               detector_modes: Callable[[str], list[str]] | None = None,
               static_dir: Path | None = STATIC_DIR) -> FastAPI:
    data_dir = Path(data_dir)
    db_path = data_dir / "app.db"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(db_path)
        conn = db.connect(db_path)
        try:
            db.fail_stale_jobs(conn)
            clean_imports(data_dir / "imports", conn)
        finally:
            conn.close()
        if start_worker:
            app.state.worker.start()
        yield
        app.state.worker.stop()

    app = FastAPI(title="auto-annotate", lifespan=lifespan)
    app.state.data_dir = data_dir
    app.state.db_path = db_path
    app.state.list_detectors = list_detectors or default_list_detectors
    app.state.detector_modes = detector_modes or default_detector_modes
    app.state.worker = Worker(db_path, load_pipeline=load_pipeline, load_pusher=load_pusher,
                              load_importers=load_importers)

    install_error_handlers(app)
    for module in (projects, images, imports, classes, jobs, annotations, models):
        app.include_router(module.router, prefix="/api")

    # Endpoint 14 belongs to the export session; include it once it exists.
    if importlib.util.find_spec("app.api.export") is not None:
        export = importlib.import_module("app.api.export")
        app.include_router(export.router, prefix="/api")

    # Static UI last, so /api/... always wins.
    if static_dir is not None and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


app = create_app()
