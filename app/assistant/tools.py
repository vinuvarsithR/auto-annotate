"""
tools.py - the Claude connector tool layer (docs/CONTRACT.md "Claude connector", "Tool layer").

One entry point, call_tool(ctx, name, arguments) -> envelope, used by BOTH front-ends (MCP server
and sidebar chat). No HTTP self-calls: each tool calls the same Python code as its API route
(route functions with a DB connection and the app's state), so checks, defaults, status rules and
error codes are the API's.

Envelope, always exactly one of:
    {"ok": true,  "data": ...}
    {"ok": false, "error": {"code", "message"}}
    {"ok": false, "needs_confirmation": {"tool", "effect", "details", "arguments"}}

Order: input check (JSON Schema) -> the entry's own checks (404/400/409) -> risky without
confirm -> needs_confirmation -> run. A call that would fail returns the error, never a confirmation.

Every write tool call (any outcome) adds one assistant_actions row. Tools that set
annotations.reviewer set it to "claude:<name>".
"""

from __future__ import annotations

import io
import json
import logging
import re
import sqlite3
import zipfile
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import best_match
from pydantic import ValidationError

from app.api import annotations as annotations_api
from app.api import classes as classes_api
from app.api import hook as hook_api
from app.api import images as images_api
from app.api import jobs as jobs_api
from app.api import models as models_api
from app.api import projects as projects_api
from app.api.deps import require_project
from app.api.errors import ApiError, not_found
from app.assistant import downloads
from app.assistant.settings import scrub
from app.assistant.tool_defs import TOOLS
from app.db import db

log = logging.getLogger("app.assistant")

TOOL_BY_NAME: dict[str, dict] = {t["name"]: t for t in TOOLS}
READ_TOOLS = tuple(t["name"] for t in TOOLS if t["kind"] == "read")
WRITE_TOOLS = tuple(t["name"] for t in TOOLS if t["kind"] == "write")
_VALIDATORS = {t["name"]: Draft202012Validator(t["input_schema"]) for t in TOOLS}

MAX_RESULT_CHARS = 50_000
DETECT_CONFIRM_ABOVE = 50   # run_detection / rerun_examples: more images than this need confirm
STATUS_CONFIRM_ABOVE = 10   # set_annotation_status: more boxes than this need confirm

_NAME = re.compile(r"^[A-Za-z0-9 ._-]{1,40}$")
NO_NAME_MESSAGE = ("set X-Autolabel-User to your name (1-40 letters, digits, space, . _ -) in "
                   "your MCP config; write tools need it")


def clean_name(raw: Any) -> str | None:
    """The name rule: 1-40 letters, digits, space, '.', '_', '-', trimmed. None if it fails."""
    if not isinstance(raw, str):
        return None
    name = raw.strip()
    return name if _NAME.match(name) else None


@dataclass
class ToolContext:
    state: Any                 # the FastAPI app.state (db_path, data_dir, worker, detectors, ...)
    channel: str               # "mcp" | "chat"
    user: str | None = None    # already cleaned with clean_name; None = no (valid) name
    button: bool = False       # chat: run by the Confirm button (#33)

    @property
    def actor(self) -> str:
        return f"claude:{self.user}" if self.user else "claude:?"

    def secrets(self) -> list[str]:
        settings = getattr(self.state, "assistant_settings", None)
        return settings.secrets() if settings is not None else []


class NeedsConfirmation(Exception):
    def __init__(self, effect: str, details: dict):
        super().__init__(effect)
        self.effect, self.details = effect, details


@dataclass
class _Run:
    """What a tool handler gets; it fills in project_id / summary for the audit row."""
    ctx: ToolContext
    conn: sqlite3.Connection
    args: dict
    project_id: int | None = None
    summary: dict | None = field(default=None)

    @property
    def confirm(self) -> bool:
        return self.args.get("confirm") is True

    @property
    def request(self):
        """Stands in for the FastAPI Request the route functions read app.state from."""
        return SimpleNamespace(app=SimpleNamespace(state=self.ctx.state))


def ok(data: Any) -> dict:
    return {"ok": True, "data": data}


def error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def outcome_of(envelope: dict) -> str:
    if envelope.get("ok"):
        return "ok"
    return "needs_confirmation" if "needs_confirmation" in envelope else "error"


def _input_error(name: str, args: Any) -> str | None:
    err = best_match(_VALIDATORS[name].iter_errors(args))
    if err is None:
        return None
    where = ".".join(str(p) for p in err.absolute_path)
    return f"{where}: {err.message}" if where else err.message


def _pydantic_message(exc: ValidationError) -> str:
    parts = []
    for e in exc.errors():
        where = ".".join(str(p) for p in e.get("loc", ()))
        msg = str(e.get("msg", "invalid")).removeprefix("Value error, ")
        parts.append(f"{where}: {msg}" if where else msg)
    return "; ".join(parts) or "invalid input"


# ------------------------------------------------------------------ entry point

def call_tool(ctx: ToolContext, name: str, arguments: Any) -> dict:
    """Run one tool. Never raises: every failure is an `error` envelope."""
    if name not in TOOL_BY_NAME:
        return error("bad_request", f"unknown tool '{name}'")
    args = {} if arguments is None else arguments
    is_write = name in WRITE_TOOLS
    conn = db.connect(ctx.state.db_path)
    run = _Run(ctx, conn, args if isinstance(args, dict) else {})
    try:
        envelope = _dispatch(run, name, args, is_write)
        envelope = _limit_size(run, name, envelope)
        if is_write:
            _log_action(run, name, args, envelope)
        return envelope
    finally:
        conn.close()


def _dispatch(run: _Run, name: str, args: Any, is_write: bool) -> dict:
    if is_write and not run.ctx.user:
        return error("bad_request", NO_NAME_MESSAGE)
    problem = _input_error(name, args)
    if problem is not None:
        return error("bad_request", problem)
    try:
        return ok(HANDLERS[name](run))
    except NeedsConfirmation as nc:
        arguments = {k: v for k, v in args.items() if k != "confirm"}
        return {"ok": False, "needs_confirmation": {
            "tool": name, "effect": nc.effect, "details": nc.details, "arguments": arguments}}
    except ApiError as e:
        return error(e.code, e.message)
    except ValidationError as e:
        return error("bad_request", _pydantic_message(e))
    except Exception as e:  # a bug or a disk problem: one scrubbed line, no traceback
        if run.conn.in_transaction:
            run.conn.rollback()
        msg = scrub(f"{type(e).__name__}: {e}".splitlines()[0][:300], run.ctx.secrets())
        log.error("assistant tool %s failed: %s", name, msg)
        return error("internal_error", msg)


def _limit_size(run: _Run, name: str, envelope: dict) -> dict:
    if len(json.dumps(envelope)) <= MAX_RESULT_CHARS:
        return envelope
    if name in WRITE_TOOLS and envelope.get("ok"):
        # the write already happened: say so, keep the audit row 'ok'
        run.summary = {**(run.summary or {}), "result_too_large": True}
        return error("bad_request", "result too large to return, but the action WAS done; "
                     "use the read tools (offset/limit) to look at the result")
    return error("bad_request", "result too large; use offset/limit")


def _log_action(run: _Run, name: str, args: Any, envelope: dict) -> None:
    outcome = outcome_of(envelope)
    if run.summary and run.summary.get("result_too_large"):
        outcome = "ok"
    if outcome == "error":
        summary = {"error": envelope["error"]["code"]}
    elif outcome == "ok":
        summary = run.summary
    else:
        summary = None
    logged = args
    if run.ctx.channel == "chat" and isinstance(args, dict):
        logged = {k: v for k, v in args.items() if k != "confirm"}
    secrets = run.ctx.secrets()
    pid = run.project_id
    if pid is not None and not run.conn.execute("SELECT 1 FROM projects WHERE id=?",
                                                (pid,)).fetchone():
        pid = None
    confirmed = 1 if (run.ctx.button or (run.ctx.channel == "mcp" and run.confirm)) else 0
    with run.conn:
        run.conn.execute(
            "INSERT INTO assistant_actions (created_at, channel, actor, project_id, tool, "
            "arguments, outcome, confirmed, summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (db.now(), run.ctx.channel, scrub(run.ctx.actor, secrets), pid, name,
             scrub(json.dumps(logged), secrets), outcome, confirmed,
             scrub(json.dumps(summary), secrets) if summary is not None else None))


# ------------------------------------------------------------------ read tools

def _list_projects(run: _Run):
    return projects_api.list_projects(run.conn)


def _project_summary(run: _Run):
    pid = run.args["project_id"]
    conn = run.conn
    project = db.project_dict(conn, require_project(conn, pid))
    by_review = {s: 0 for s in db.REVIEW_STATUSES}
    for r in conn.execute("SELECT review_status, COUNT(*) AS n FROM images WHERE project_id=? "
                          "GROUP BY review_status", (pid,)):
        by_review[r["review_status"]] = r["n"]
    boxes = {r["name"]: db.zero_counts() for r in conn.execute(
        "SELECT name FROM classes WHERE project_id=? ORDER BY id", (pid,))}
    for r in conn.execute(
            "SELECT c.name, a.status, COUNT(*) AS n FROM annotations a "
            "JOIN classes c ON c.id = a.class_id JOIN images i ON i.id = a.image_id "
            "WHERE i.project_id=? GROUP BY c.name, a.status", (pid,)):
        boxes.setdefault(r["name"], db.zero_counts())[r["status"]] = r["n"]
    latest = conn.execute("SELECT * FROM jobs WHERE project_id=? ORDER BY id DESC LIMIT 1",
                          (pid,)).fetchone()
    return {"project": project, "images_by_review_status": by_review, "boxes_by_class": boxes,
            "latest_job": db.job_dict(latest) if latest is not None else None}


def _list_images(run: _Run):
    a = run.args
    return images_api.list_images(a["project_id"], a.get("status"), a.get("review_status"),
                                  a.get("offset", 0), a.get("limit", 50), run.conn)


def _get_annotations(run: _Run):
    return annotations_api.list_annotations(run.args["image_id"], run.args.get("status"), run.conn)


def _list_classes(run: _Run):
    return classes_api.list_classes(run.args["project_id"], run.conn)


def _get_job(run: _Run):
    return jobs_api.get_job(run.args["job_id"], run.conn)


def _list_models(run: _Run):
    return models_api.list_models(run.request, project_id=run.args["project_id"])


def _get_hook(run: _Run):
    return hook_api.get_hook(run.args["project_id"], run.conn)


# ------------------------------------------------------------------ write tools

def _add_class(run: _Run):
    a, conn = run.args, run.conn
    pid = run.project_id = a["project_id"]
    require_project(conn, pid)
    name = a["name"].strip()
    if not name:
        raise ApiError(400, "bad_request", "class name must not be empty")
    rows = conn.execute("SELECT * FROM classes WHERE project_id=? ORDER BY id", (pid,)).fetchall()
    if any(r["name"] == name for r in rows):
        raise ApiError(400, "bad_request", f"class '{name}' already exists in project {pid}")
    description = (a.get("description") or "").strip() or name  # default = the name
    new = classes_api.ClassIn(name=name, description=description,
                              min_score=a.get("min_score"), accept_score=a.get("accept_score"))
    keep = [classes_api.ClassIn(name=r["name"], description=r["description"],
                                min_score=r["min_score"], accept_score=r["accept_score"])
            for r in rows]
    result = classes_api.replace_classes(pid, classes_api.ClassesIn(classes=keep + [new]), conn)
    created = next(c for c in result["classes"] if c["name"] == name)
    run.summary = {"class_id": created["id"]}
    return {"class": created, "classes": result["classes"]}


def _job_effect(model: str, mode, masks: str, n: int, prefix: str, which: str = "") -> str:
    return (f"{prefix} with {model} (mode {mode or 'none'}, masks {masks}) on {n} images{which}. "
            "Earlier unreviewed model boxes of the classes that run are replaced on those images.")


def _run_detection(run: _Run):
    a, conn = run.args, run.conn
    pid = run.project_id = a["project_id"]
    require_project(conn, pid)
    fields: dict = {"model": a["model"], "options": {}}
    for key in ("mode", "image_ids", "masks"):
        if key in a:
            fields[key] = a[key]
    total, config = jobs_api.plan_job(pid, jobs_api.JobIn(**fields), run.request, conn)
    masks = config.get("masks", "none")
    if total > DETECT_CONFIRM_ABOVE and not run.confirm:
        raise NeedsConfirmation(
            _job_effect(config["model"], config["mode"], masks, total, "Start a detect job"),
            {"images": total, "model": config["model"], "mode": config["mode"], "masks": masks})
    job = jobs_api.insert_job(run.request, conn, pid, total, config)
    run.summary = {"job_id": job["id"]}
    return job


def _rerun_examples(run: _Run):
    conn = run.conn
    pid = run.project_id = run.args["project_id"]
    require_project(conn, pid)
    total, config = jobs_api.plan_rerun(pid, jobs_api.JobIn(rerun=True), run.request, conn)
    masks = config.get("masks", "none")
    if total > DETECT_CONFIRM_ABOVE and not run.confirm:
        raise NeedsConfirmation(
            _job_effect(config["model"], config["mode"], masks, total,
                        f"Re-run examples-mode job {config['rerun_of']}",
                        " still in review_status todo"),
            {"images": total, "model": config["model"], "mode": config["mode"], "masks": masks})
    job = jobs_api.insert_job(run.request, conn, pid, total, config)
    run.summary = {"job_id": job["id"]}
    return job


def _set_annotation_status(run: _Run):
    a, conn = run.args, run.conn
    ids: list[int] = a["annotation_ids"]
    status = "accepted" if a["action"] == "approve" else "rejected"
    rows = []
    for aid in ids:
        row = db.get_annotation(conn, aid)
        if row is None:
            raise not_found("annotation", aid)
        rows.append(row)
    run.project_id = _project_of_image(conn, rows[0]["image_id"])
    if len(ids) > STATUS_CONFIRM_ABOVE and not run.confirm:
        by_class: dict[str, int] = {}
        for r in rows:
            by_class[r["class_name"]] = by_class.get(r["class_name"], 0) + 1
        verb = "Approve" if a["action"] == "approve" else "Reject"
        raise NeedsConfirmation(f"{verb} {len(ids)} boxes (set status {status}).",
                                {"boxes": len(ids), "action": a["action"], "by_class": by_class})
    t = db.now()
    conn.execute("BEGIN IMMEDIATE")  # all or nothing
    try:
        for aid in ids:  # #13 with body {status, reviewer}: explicit status wins
            cur = conn.execute("UPDATE annotations SET status=?, reviewer=?, updated_at=? "
                               "WHERE id=?", (status, run.ctx.actor, t, aid))
            if cur.rowcount == 0:  # deleted since the check above
                raise not_found("annotation", aid)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    run.summary = {"annotation_ids": ids, "status": status}
    return {"annotations": [db.annotation_dict(db.get_annotation(conn, aid)) for aid in ids]}


def _project_of_image(conn: sqlite3.Connection, iid: int) -> int | None:
    row = conn.execute("SELECT project_id FROM images WHERE id=?", (iid,)).fetchone()
    return row[0] if row is not None else None


UPDATE_FIELDS = ("bbox", "polygon", "class_name", "status")


def _update_annotation(run: _Run):
    a, conn = run.args, run.conn
    aid = a["annotation_id"]
    sent = {k: a[k] for k in UPDATE_FIELDS if k in a}
    if not sent:
        raise ApiError(400, "bad_request", "send at least one of " + ", ".join(UPDATE_FIELDS))
    patch = {k: v for k, v in sent.items() if k != "class_name"}
    annotations_api.AnnotationPatch(**patch)  # body check first, as the route does
    row = db.get_annotation(conn, aid)
    if row is None:
        raise not_found("annotation", aid)
    pid = run.project_id = _project_of_image(conn, row["image_id"])
    if "class_name" in sent:
        cls = conn.execute("SELECT id FROM classes WHERE project_id=? AND name=?",
                           (pid, sent["class_name"])).fetchone()
        if cls is None:
            raise ApiError(400, "bad_class_id",
                           f"class '{sent['class_name']}' is not in project {pid}")
        patch["class_id"] = cls["id"]
    patch["reviewer"] = run.ctx.actor
    data = annotations_api.patch_annotation(aid, annotations_api.AnnotationPatch(**patch), conn)
    run.summary = {"annotation_id": aid}
    return data


def _delete_annotation(run: _Run):
    conn = run.conn
    aid = run.args["annotation_id"]
    row = db.get_annotation(conn, aid)
    if row is None:
        raise not_found("annotation", aid)
    run.project_id = _project_of_image(conn, row["image_id"])
    if not run.confirm:
        raise NeedsConfirmation(
            f"Permanently delete box {aid} ({row['class_name']}, source {row['source']}, "
            f"status {row['status']}) on image {row['image_id']}. This cannot be undone.",
            {"annotation_id": aid, "image_id": row["image_id"], "class_name": row["class_name"],
             "source": row["source"], "status": row["status"]})
    annotations_api.delete_annotation(aid, conn)
    run.summary = {"deleted": aid}
    return {"deleted": aid}


def _set_hook_point(run: _Run):
    a, conn = run.args, run.conn
    pid = run.project_id = _project_of_image(conn, a["image_id"])
    if pid is None:  # #28: an image not in the project -> bad_image_ids
        raise ApiError(400, "bad_image_ids", f"image {a['image_id']} not found")
    data = hook_api.put_hook_point(pid, {"image_id": a["image_id"], "x": a["x"], "y": a["y"]},
                                   conn)
    run.summary = {"sequence": data["sequence"]}
    return data


_FILENAME = re.compile(r'filename="([^"]+)"')


def _make_train_package(run: _Run):
    from app.api import export as export_api  # export's module: imported, never edited

    a = run.args
    pid = run.project_id = a["project_id"]
    task = a.get("task", "detect")
    response = export_api.train_package(pid, {"task": task}, run.conn)
    blob = bytes(response.body)
    m = _FILENAME.search(response.headers.get("content-disposition", ""))
    file_name = m.group(1) if m else f"project_{pid}_train_{task}.zip"
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        split = json.loads(zf.read("split.json"))
    token, expires_at = downloads.save(run.ctx.state.data_dir, blob, file_name)
    run.summary = {"file_name": file_name, "size_bytes": len(blob)}
    return {"download_url": f"/api/assistant/downloads/{token}", "file_name": file_name,
            "size_bytes": len(blob), "expires_at": expires_at, "split": split}


def _push_to_roboflow(run: _Run):
    from app.api import export as export_api  # export's module: imported, never edited

    a, conn = run.args, run.conn
    pid = run.project_id = a["project_id"]
    require_project(conn, pid)
    # the #21 checks, in the route's order, without creating the job
    try:
        statuses = export_api.validate_statuses(a.get("statuses"))
    except ValueError as e:
        raise ApiError(400, "bad_request", str(e))
    settings = export_api.roboflow_settings()
    if isinstance(settings, list):  # names only, never values
        raise ApiError(409, "conflict",
                       "Roboflow settings missing from .env: " + ", ".join(settings))
    if not conn.execute("SELECT 1 FROM images WHERE project_id=? AND review_status IN "
                        "('done', 'empty') LIMIT 1", (pid,)).fetchone():
        raise ApiError(409, "conflict", f"project {pid} has no done or empty images to push")
    total = conn.execute(
        "SELECT COUNT(*) FROM images WHERE project_id=? AND review_status IN ('done', 'empty') "
        "AND roboflow_image_id IS NULL", (pid,)).fetchone()[0]
    if not run.confirm:
        plural = "image" if total == 1 else "images"
        raise NeedsConfirmation(
            f"Upload {total} {plural} (review_status done or empty, not pushed yet) and their "
            f"boxes with status {', '.join(statuses)} to the Roboflow project set in the "
            "server's .env.",
            {"images": total, "statuses": statuses})
    job = export_api.push_project(pid, run.request, export_api.PushIn(statuses=a.get("statuses")),
                                  conn)
    run.summary = {"job_id": job["id"]}
    return job


HANDLERS: dict[str, Callable[[_Run], Any]] = {
    "list_projects": _list_projects,
    "project_summary": _project_summary,
    "list_images": _list_images,
    "get_annotations": _get_annotations,
    "list_classes": _list_classes,
    "get_job": _get_job,
    "list_models": _list_models,
    "get_hook": _get_hook,
    "add_class": _add_class,
    "run_detection": _run_detection,
    "rerun_examples": _rerun_examples,
    "set_annotation_status": _set_annotation_status,
    "update_annotation": _update_annotation,
    "delete_annotation": _delete_annotation,
    "set_hook_point": _set_hook_point,
    "make_train_package": _make_train_package,
    "push_to_roboflow": _push_to_roboflow,
}
assert list(HANDLERS) == [t["name"] for t in TOOLS], "HANDLERS must follow tool_defs order"
