"""
main.py - the FastAPI app.

Run it (from the repo root, inside .venv):
    uvicorn app.main:app --port 8000

* API under /api (docs/CONTRACT.md section 2), UI from app/static at /.
* data/app.db and data/uploads/ are created on startup.
* A background worker thread runs detect, roboflow_push and import jobs (app/worker/worker.py).
* Claude connector (app/assistant/): #31 - #35 under /api/assistant, the MCP server at /mcp/
  (token / loopback gate), settings from .env read in the lifespan.
* create_app(...) lets tests use a temp folder, a fake detector and no thread.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import annotations, classes, hook, images, imports, jobs, models, projects, weights
from app.api.errors import install_error_handlers
from app.assistant import routes as assistant_routes
from app.assistant.mcp_server import McpGate
from app.db import db
from app.worker.worker import ImporterLoader, PipelineLoader, PusherLoader, Worker

STATIC_DIR = Path(__file__).parent / "static"


def start_assistant(app: FastAPI, anthropic_client=None):
    """Claude connector at startup: settings from .env, old downloads deleted, the chat manager,
    and a NEW MCP Server for this app. Returns the Server (its session manager is run by the
    caller), or None when the mcp package is missing (then /mcp/ answers 503)."""
    from app.assistant import downloads
    from app.assistant.chat import ChatManager
    from app.assistant.settings import load_settings

    settings = load_settings()
    app.state.assistant_settings = settings
    downloads.cleanup(app.state.data_dir)

    def make_client():
        if anthropic_client is not None:
            return anthropic_client
        import anthropic
        return anthropic.Anthropic(api_key=settings.api_key)  # the only credential used

    app.state.assistant_chat = ChatManager(app.state, settings, make_client)
    try:
        from app.assistant.mcp_server import build_http_app, build_server
        server = build_server(app.state)
        app.state.mcp_http_app = build_http_app(server, settings.token_required)
    except ImportError:
        logging.getLogger("app").warning("mcp package not installed: /mcp/ is off")
        app.state.mcp_http_app = None
        return None
    return server


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


# Train on Colab (#24 - #26): the models session's functions, imported only when used.
def default_inspect_weights(path: str, model=None):
    from autolabel.models import inspect_weights
    return inspect_weights(path, model=model)


def default_register_trained(name: str, path: str, task: str, model=None) -> None:
    from autolabel.models import register_trained
    register_trained(name, path, task, model=model)


def default_unregister_trained(name: str) -> None:
    try:
        from autolabel.models import unregister_trained
    except ImportError:  # models code not installed: nothing was ever registered
        return
    unregister_trained(name)


def create_app(data_dir: str | Path = "data", *, start_worker: bool = True,
               load_pipeline: PipelineLoader | None = None,
               load_pusher: PusherLoader | None = None,
               load_importers: ImporterLoader | None = None,
               list_detectors: Callable[[], list[str]] | None = None,
               detector_modes: Callable[[str], list[str]] | None = None,
               static_dir: Path | None = STATIC_DIR,
               inspect_weights: Callable | None = None,
               register_trained: Callable | None = None,
               unregister_trained: Callable | None = None,
               anthropic_client=None) -> FastAPI:
    """anthropic_client: None = built from ANTHROPIC_API_KEY on the first chat message;
    tests pass a fake with .messages.create and .beta.messages.create."""
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
        mcp_server = start_assistant(app, anthropic_client)
        if start_worker:
            app.state.worker.start()
        try:
            if mcp_server is not None:
                async with mcp_server.session_manager.run():  # a session manager runs only once
                    yield
            else:
                yield
        finally:
            app.state.worker.stop()

    app = FastAPI(title="auto-annotate", lifespan=lifespan)
    app.state.data_dir = data_dir
    app.state.db_path = db_path
    app.state.list_detectors = list_detectors or default_list_detectors
    app.state.detector_modes = detector_modes or default_detector_modes
    app.state.inspect_weights = inspect_weights or default_inspect_weights
    app.state.register_trained = register_trained or default_register_trained
    app.state.unregister_trained = unregister_trained or default_unregister_trained
    app.state.worker = Worker(db_path, load_pipeline=load_pipeline, load_pusher=load_pusher,
                              load_importers=load_importers,
                              register_trained=app.state.register_trained)

    install_error_handlers(app)
    for module in (projects, images, imports, classes, jobs, annotations, models, weights, hook,
                   assistant_routes):
        app.include_router(module.router, prefix="/api")
    app.mount("/mcp", McpGate(app.state), name="mcp")  # #36; not under /api

    # Endpoint 14 belongs to the export session; include it once it exists.
    if importlib.util.find_spec("app.api.export") is not None:
        export = importlib.import_module("app.api.export")
        app.include_router(export.router, prefix="/api")

    # Static UI last, so /api/... always wins.
    if static_dir is not None and Path(static_dir).is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app


app = create_app()
