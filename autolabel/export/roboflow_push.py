"""
roboflow_push.py - the Roboflow part of CONTRACT #21, using the roboflow Python SDK
(never the Roboflow MCP connector).

    settings, missing = roboflow_settings()      # reads .env; missing = names, never values
    project = open_project(settings)             # once per job
    rf_id = push_image(project, image, path, classes, annotations, batch_name(), statuses)

The key is never printed, logged or put into an error: SDK output is swallowed and every
SDK error becomes a one-line PushError with the key replaced by `***`.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Mapping

from autolabel.export.formats import to_coco
from autolabel.export.select import validate_statuses

log = logging.getLogger(__name__)

ENV_NAMES = ("ROBOFLOW_API_KEY", "ROBOFLOW_WORKSPACE", "ROBOFLOW_PROJECT")
REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_ERROR_LEN = 300


class PushError(Exception):
    """One-line, key-free error from a push (goes into the job's `error`)."""


@dataclass(frozen=True)
class RoboflowSettings:
    workspace: str
    project: str
    api_key: str = field(repr=False)   # repr=False: never shows up when printed


def roboflow_settings(
    env: Mapping[str, str] | None = None, dotenv_path: Path | None = None
) -> tuple[RoboflowSettings | None, list[str]]:
    """
    Read the 3 settings. `env` defaults to os.environ after loading `.env` from the repo
    root (existing environment variables win). Returns (settings, []) or (None, missing
    names). Only names are reported, never values.
    """
    if env is None:
        try:
            from dotenv import load_dotenv

            load_dotenv(dotenv_path or REPO_ROOT / ".env", override=False)
        except ImportError:  # python-dotenv is in requirements; plain env vars still work
            pass
        env = os.environ
    values = {n: (env.get(n) or "").strip() for n in ENV_NAMES}
    missing = [n for n in ENV_NAMES if not values[n]]
    if missing:
        return None, missing
    return (
        RoboflowSettings(
            workspace=values["ROBOFLOW_WORKSPACE"],
            project=values["ROBOFLOW_PROJECT"],
            api_key=values["ROBOFLOW_API_KEY"],
        ),
        [],
    )


def batch_name(now: datetime | None = None) -> str:
    """`autolabel_<YYYY-MM-DD>` for the job start (UTC)."""
    now = now or datetime.now(timezone.utc)
    return f"autolabel_{now.astimezone(timezone.utc):%Y-%m-%d}"


def safe_message(exc: BaseException, secrets: list[str]) -> str:
    """One line, secrets replaced by ***, at most MAX_ERROR_LEN chars."""
    msg = f"{type(exc).__name__}: {exc}"
    for s in secrets:
        if s:
            msg = msg.replace(s, "***")
    msg = " ".join(msg.split())
    return msg[:MAX_ERROR_LEN]


@contextlib.contextmanager
def _quiet_sdk(secrets: list[str], what: str):
    """Swallow SDK stdout/stderr; turn any SDK error into a key-free PushError."""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield
    except PushError:
        raise
    except Exception as e:
        # `from None` drops the original exception (and its message) from the chain.
        raise PushError(f"{what} failed: {safe_message(e, secrets)}") from None


def open_project(settings: RoboflowSettings):
    """Open the Roboflow project with the SDK (one network call; do it once per job)."""
    with _quiet_sdk([settings.api_key], "opening Roboflow project"):
        from roboflow import Roboflow

        rf = Roboflow(api_key=settings.api_key)
        return rf.workspace(settings.workspace).project(settings.project)


def push_image(
    project,
    image: dict,
    path: str | os.PathLike,
    classes: list[dict],
    annotations: list[dict],
    batch: str,
    statuses: list[str] | None = None,
    secrets: list[str] | None = None,
) -> str:
    """
    Upload one image and its boxes; return the Roboflow image id.

    - `project`: from open_project (an SDK Project).
    - `image`: Image JSON; must be review_status done or empty. If it already has a
      `roboflow_image_id`, nothing is uploaded and that id is returned.
    - `path`: the image file on disk. It is uploaded under its original `file_name`.
    - boxes follow the #14 rules (same `statuses`, never `rejected`); an `empty` image
      is uploaded with a COCO file that has no boxes.
    - `secrets`: strings to scrub from errors (pass the API key).
    Raises PushError (one line, key-free) on any failure.
    """
    if image.get("roboflow_image_id"):
        return str(image["roboflow_image_id"])
    if image.get("review_status") not in ("done", "empty"):
        raise PushError(f"image {image['id']} is not done/empty")
    statuses = validate_statuses(statuses)
    secrets = list(secrets or [])

    coco = to_coco([image], classes, annotations, statuses)
    coco["images"][0]["file_name"] = image["file_name"]
    stem = PurePosixPath(image["file_name"]).stem
    ann = {"name": f"{stem}.json", "rawText": json.dumps(coco)}

    with tempfile.TemporaryDirectory() as tmp:
        # Roboflow names the image after the uploaded file; the file on disk is
        # `<id>_<file_name>`, so copy it to its original name first.
        upload_path = Path(tmp) / image["file_name"]
        shutil.copyfile(path, upload_path)
        with _quiet_sdk(secrets, f"uploading image {image['id']}"):
            result = project.single_upload(
                image_path=str(upload_path),
                annotation_path=ann,
                batch_name=batch,
                annotation_overwrite=True,
            )

    rf_id = ((result or {}).get("image") or {}).get("id")
    if not rf_id:
        raise PushError(f"uploading image {image['id']} failed: Roboflow returned no image id")
    log.info("pushed image %s to Roboflow as %s", image["id"], rf_id)
    return str(rf_id)
