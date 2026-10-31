"""
main.py - the FastAPI app.

Run it (from the repo root, inside .venv):
    uvicorn app.main:app --port 8000

* API under /api (docs/CONTRACT.md section 2), UI from app/static at /.
* data/app.db and data/uploads/ are created on startup.
* A background worker thread runs detect jobs (app/worker/worker.py).
* create_app(...) lets tests use a temp folder, a fake detector and no thread.
"""

from __future__ import annotations

import importlib
import importlib.util
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import annotations, classes, images, jobs, projects
from app.api.errors import install_error_handlers
from app.db import db
from app.worker.worker import PipelineLoader, Worker

STATIC_DIR = Path(__file__).parent / "static"


def default_list_detectors() -> list[str]:
    from autolabel.models import list_detectors
    return list(list_detectors())


def create_app(data_dir: str | Path = "data", *, start_worker: bool = True,
               load_pipeline: PipelineLoader | None = None,
               list_detectors: Callable[[], list[str]] | None = None,
               static_dir: Path | None = STATIC_DIR) -> FastAPI:
    data_dir = Path(data_dir)
    db_path = data_dir / "app.db"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init_db(db_path)
        conn = db.connect(db_path)
        try:
            db.fail_stale_jobs(conn)
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
    app.state.worker = Worker(db_path, load_pipeline=load_pipeline)

    install_error_handlers(app)
    for module in (projects, images, classes, jobs, annotations):
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
