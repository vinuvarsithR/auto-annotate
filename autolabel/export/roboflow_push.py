"""
roboflow_push.py - the Roboflow part of CONTRACT #21, using the roboflow Python SDK
(never the Roboflow MCP connector).

The backend worker does the loop (which images, skipping pushed ones, saving ids).
This module only provides:

    settings = roboflow_settings()     # RoboflowSettings, or the list of missing .env names
    rf_id = push_image(settings, image, boxes, class_names, batch_name)

`image` = {"id", "file_name", "path", "width", "height"}; `boxes` = [{"class_name", "bbox"}]
(already filtered by the worker; an empty list = image with no labels).

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
from pathlib import Path, PurePosixPath
from typing import Mapping

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
) -> RoboflowSettings | list[str]:
    """
    Read the 3 settings. `env` defaults to os.environ after loading `.env` from the repo
    root (existing environment variables win). Returns RoboflowSettings, or the list of
    missing names (names only, never values).
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
        return missing
    return RoboflowSettings(
        workspace=values["ROBOFLOW_WORKSPACE"],
        project=values["ROBOFLOW_PROJECT"],
        api_key=values["ROBOFLOW_API_KEY"],
    )


def safe_message(exc: BaseException, secrets: list[str]) -> str:
    """One line, secrets replaced by ***, at most MAX_ERROR_LEN chars."""
    msg = f"{type(exc).__name__}: {exc}"
    for s in secrets:
        if s:
            msg = msg.replace(s, "***")
    return " ".join(msg.split())[:MAX_ERROR_LEN]


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


# One SDK project per (workspace, project), so a push job opens it once, not per image.
_projects: dict[tuple[str, str], object] = {}


def _project(settings: RoboflowSettings):
    k = (settings.workspace, settings.project)
    if k not in _projects:
        with _quiet_sdk([settings.api_key], "opening Roboflow project"):
            from roboflow import Roboflow

            rf = Roboflow(api_key=settings.api_key)
            _projects[k] = rf.workspace(settings.workspace).project(settings.project)
    return _projects[k]


# The Roboflow backend only recognises a COCO file that has at least one annotation. For
# an image with no boxes, the SDK's own COCO importer (roboflow.util.folderparser,
# _filterIndividualAnnotations, SDK 1.7.3) adds this placeholder, which points at an image id
# that does not exist, so the uploaded image is saved as a null (no-object) image.
NULL_PLACEHOLDER = {"id": 999999999, "image_id": 999999999, "category_id": 0,
                    "area": 1, "segmentation": [], "iscrowd": 0}
ANNOTATION_NAME = "annotation.coco.json"   # same name the SDK importer uses


def coco_for_image(image: dict, boxes: list[dict], class_names: list[str]) -> dict:
    """
    One-image COCO dict; categories 1-based in class_names order. No boxes -> only the
    NULL_PLACEHOLDER annotation (Roboflow then marks the image as null).
    """
    idx = {n: i for i, n in enumerate(class_names)}
    anns = []
    for i, b in enumerate(boxes, start=1):
        if b["class_name"] not in idx:
            raise PushError(f"image {image['id']}: unknown class {b['class_name']!r}")
        x, y, w, h = (float(v) for v in b["bbox"])
        anns.append({"id": i, "image_id": 1, "category_id": idx[b["class_name"]] + 1,
                     "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0, "segmentation": []})
    return {
        "info": {},
        "licenses": [],
        "images": [{"id": 1, "file_name": image["file_name"],
                    "width": image["width"], "height": image["height"]}],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(class_names)],
        "annotations": anns or [dict(NULL_PLACEHOLDER)],
    }


def push_image(
    settings: RoboflowSettings,
    image: dict,
    boxes: list[dict],
    class_names: list[str],
    batch_name: str,
) -> str:
    """
    Upload one image with its boxes to Roboflow; return the Roboflow image id.
    The file at image["path"] is uploaded under its original image["file_name"], with a
    one-image COCO annotation (an empty list uploads a null image). Raises PushError (one line,
    key-free) on any failure.
    """
    secrets = [settings.api_key]
    project = _project(settings)
    coco = coco_for_image(image, boxes, class_names)
    ann = {"name": ANNOTATION_NAME, "rawText": json.dumps(coco)}

    with tempfile.TemporaryDirectory() as tmp:
        # Roboflow names the image after the uploaded file; the file on disk is
        # `<id>_<file_name>`, so copy it to its original name first.
        upload_path = Path(tmp) / PurePosixPath(image["file_name"]).name
        try:
            shutil.copyfile(image["path"], upload_path)
        except OSError as e:
            raise PushError(f"image {image['id']}: file not readable ({type(e).__name__})") from None
        with _quiet_sdk(secrets, f"uploading image {image['id']}"):
            result = project.single_upload(
                image_path=str(upload_path),
                annotation_path=ann,
                batch_name=batch_name,
                annotation_overwrite=True,
            )

    rf_id = ((result or {}).get("image") or {}).get("id")
    if not rf_id:
        raise PushError(f"uploading image {image['id']} failed: Roboflow returned no image id")
    log.info("pushed image %s to Roboflow as %s", image["id"], rf_id)
    return str(rf_id)
