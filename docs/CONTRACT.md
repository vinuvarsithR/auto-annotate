# CONTRACT.md - what the 4 parallel sessions build against

Sessions: **backend**, **models**, **ui**, **export**, plus the **integrator** (the main
session). If something here is wrong or missing, write it in `docs/agents/<name>.md`
under "Contract change requests". Only the integrator edits this file.

Conventions used everywhere:

| Thing | Rule |
|---|---|
| IDs | integers, assigned by SQLite |
| Times | ISO-8601 UTC text, e.g. `"2026-10-30T14:05:00Z"` |
| Boxes | `[x, y, w, h]` in pixels, top-left origin (COCO style, same as `autolabel/evaluate.Box`) |
| Masks | COCO RLE `{"size": [h, w], "counts": "<compressed string>"}` or `null` |
| Classes in Python | by **name** (string), as in `Box.category` |
| Classes in API/DB | by `class_id`, and every annotation response also includes `class_name` |
| Secrets | `ROBOFLOW_API_KEY` is read from `.env` on the server only. Never in a request, response, log or DB row. |

---

## 1. SQLite schema

File: `data/app.db` (gitignored). The backend owns `app/db/` and creates the tables on startup.

```sql
CREATE TABLE projects (
  id          INTEGER PRIMARY KEY,
  name        TEXT NOT NULL UNIQUE,
  created_at  TEXT NOT NULL
);

CREATE TABLE images (
  id           INTEGER PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  file_name    TEXT NOT NULL,             -- original upload name
  path         TEXT NOT NULL,             -- on disk: data/uploads/<project_id>/<id>_<file_name>
  width        INTEGER NOT NULL,
  height       INTEGER NOT NULL,
  uploaded_at  TEXT NOT NULL,
  review_status TEXT NOT NULL DEFAULT 'todo'
                CHECK (review_status IN ('todo', 'done', 'empty')),  -- set only by #20
  roboflow_image_id TEXT,               -- NULL = not pushed yet; set only by the #21 push job
  UNIQUE (project_id, file_name)
);

CREATE TABLE classes (
  id            INTEGER PRIMARY KEY,
  project_id    INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,            -- short label, e.g. "biax"
  description   TEXT NOT NULL,            -- text prompt, e.g. "a short yellow crossbar ..."
  min_score     REAL,                     -- NULL = model default; boxes below are not stored
  accept_score  REAL,                     -- NULL = never auto-accept; boxes >= become auto_accepted
  UNIQUE (project_id, name)
);

CREATE TABLE jobs (
  id           INTEGER PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  kind         TEXT NOT NULL CHECK (kind IN ('detect', 'roboflow_push')),
  status       TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
  progress     REAL NOT NULL DEFAULT 0,   -- 0..1 = done / total
  done         INTEGER NOT NULL DEFAULT 0,
  total        INTEGER NOT NULL DEFAULT 0,
  error        TEXT,                      -- one-line message when status = failed
  config       TEXT NOT NULL,             -- JSON: the request body of POST .../jobs
  created_at   TEXT NOT NULL,
  started_at   TEXT,
  finished_at  TEXT
);

CREATE TABLE annotations (
  id          INTEGER PRIMARY KEY,
  image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  class_id    INTEGER NOT NULL REFERENCES classes(id),
  x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  mask        TEXT,                       -- COCO RLE JSON or NULL
  score       REAL,                       -- NULL for human-drawn boxes
  source      TEXT NOT NULL,              -- 'owlv2' | 'owlv2_image' | 'gdino' | 'pipeline' | 'human'
  status      TEXT NOT NULL CHECK (status IN
                ('auto_accepted', 'needs_review', 'accepted', 'rejected', 'edited')),
  job_id      INTEGER REFERENCES jobs(id),  -- NULL for human-drawn boxes
  reviewer    TEXT,                       -- free text name for now (no login until Phase 6)
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX idx_ann_image ON annotations(image_id);
CREATE INDEX idx_ann_status ON annotations(status);
```

Status rules (the backend enforces them):

| Event | New status |
|---|---|
| Model box, `score >= accept_score` | `auto_accepted` |
| Model box, otherwise | `needs_review` |
| Human drew a new box | `accepted` (source `human`) |
| Reviewer approves | `accepted` |
| Reviewer deletes | `rejected` (the row is kept so we can measure precision later) |
| Reviewer changes box, mask or class | `edited` |
| Reviewer undoes a change | the status sent in the request |

An explicit `status` in a PATCH (#13) always wins. Only when `status` is absent does a
bbox / mask / class change make the box `edited`.

A new `detect` job on an image deletes that image's earlier **model** boxes that are
still `auto_accepted` or `needs_review`. It never touches human-reviewed rows.

Image review status (`images.review_status`):

| Value | Meaning |
|---|---|
| `todo` | not checked yet (default for new images and for rows that existed before this column) |
| `done` | a reviewer checked the image |
| `empty` | a reviewer checked it and there is nothing to label |

Only #20 changes `review_status`; nothing changes it automatically. `empty` needs every box of
the image to be `rejected` (#20 -> 409 `conflict` otherwise), and #12 on an `empty` image ->
409 `conflict`. A detect job (#9) or an undo (#13) may still add boxes to an `empty` image; it
stays `empty`. The backend adds the column on startup (`ALTER TABLE ... DEFAULT 'todo'`) if missing.

`images.roboflow_image_id` is the id Roboflow returned when the image was pushed (#21). The backend
adds it on startup (`ALTER TABLE images ADD COLUMN roboflow_image_id TEXT`) if missing. It is not
part of the Image JSON.

Export / push rules (#14 and #21):

| Rule | Value |
|---|---|
| Images | only `review_status` `done` or `empty`; `todo` images are left out |
| `empty` image | exported / pushed as an image with no labels |
| Approved statuses (the default `statuses`) | `auto_accepted`, `accepted`, `edited` |
| `statuses` allowed values | any #13 status **except** `rejected`; `rejected` or unknown -> 400 `bad_request` |
| Class order | project class order = the order #7 returns. YOLO class index = position in that list; COCO categories in the same order |

---

## 2. REST API

The base path is `/api`, all bodies are JSON, and the server is FastAPI on `http://localhost:8000`.
The UI is served at `/` from `app/static/`.

Errors always look like this, with a matching HTTP status (400, 404, 409 or 500):
```json
{"error": {"code": "not_found", "message": "image 99 not found"}}
```

Error codes (the UI may rely on these; `message` is for humans and may change):

| HTTP | `code` | When |
|---|---|---|
| 400 | `bad_request` | invalid body or query (incl. FastAPI validation), empty/duplicate names, unknown `status` |
| 400 | `unknown_model` | `model` not in `list_detectors()` |
| 400 | `no_classes` | detect job on a project with no classes |
| 400 | `bad_image_ids` | `image_ids` not all in this project |
| 400 | `bad_class_id` | `class_id` not in this image's project |
| 404 | `not_found` | unknown id, unknown path, or image file missing on disk |
| 405 | `method_not_allowed` | wrong HTTP method for the path |
| 409 | `name_taken` | project name already used (#1 create, #17 rename) |
| 409 | `class_in_use` | PUT classes would delete a class that still has annotations |
| 409 | `conflict` | any other 409, e.g. deleting a project with a queued/running job (#18), `empty` on an image with boxes (#20), a new box on an `empty` image (#12), no `done`/`empty` images (#14, #21), Roboflow settings missing from `.env` (#21) |
| 500 | `internal_error` | unexpected server error (details only in the server log) |

| # | Method | Path | Purpose |
|---|---|---|---|
| 1 | POST | `/projects` | create project |
| 2 | GET | `/projects` | list projects |
| 3 | GET | `/projects/{pid}` | project + counts |
| 4 | POST | `/projects/{pid}/images` | upload images (multipart) |
| 5 | GET | `/projects/{pid}/images` | list images |
| 6 | GET | `/images/{iid}/file` | raw image bytes |
| 7 | GET | `/projects/{pid}/classes` | list classes |
| 8 | PUT | `/projects/{pid}/classes` | replace class list |
| 9 | POST | `/projects/{pid}/jobs` | start detect job |
| 10 | GET | `/jobs/{jid}` | job status |
| 11 | GET | `/images/{iid}/annotations` | boxes for one image |
| 12 | POST | `/images/{iid}/annotations` | add human box |
| 13 | PATCH | `/annotations/{aid}` | review / edit a box |
| 14 | POST | `/projects/{pid}/export` | download COCO / YOLO zip |
| 15 | - | - | removed (replaced by #21) |
| 16 | GET | `/models` | detector names for POST .../jobs |
| 17 | PATCH | `/projects/{pid}` | rename project |
| 18 | DELETE | `/projects/{pid}` | delete project and everything in it |
| 19 | DELETE | `/annotations/{aid}` | hard-delete one box |
| 20 | PATCH | `/images/{iid}` | set image review status |
| 21 | POST | `/projects/{pid}/roboflow/push` | start Roboflow push job |

### 1. POST /projects
Request `{"name": "eot_crane"}`. Response 201 is a **Project**:
```json
{"id": 1, "name": "eot_crane", "created_at": "2026-10-30T14:00:00Z",
 "counts": {"images": 0, "auto_accepted": 0, "needs_review": 0, "accepted": 0, "rejected": 0, "edited": 0}}
```
If the name already exists, the response is 409 `name_taken`.

### 2. GET /projects
Response `{"projects": [Project, ...]}`.

### 3. GET /projects/{pid}
Response is a Project, as in #1.

### 4. POST /projects/{pid}/images
The request is `multipart/form-data` with one or more `files` fields (jpg/png).
Response 201:
```json
{"images": [Image, ...], "skipped": [{"file_name": "a.jpg", "reason": "duplicate"}]}
```
**Image**:
```json
{"id": 10, "project_id": 1, "file_name": "biax__cam0_2026-09-29_18-48-25_00290.jpg", "width": 1280, "height": 720,
 "url": "/api/images/10/file", "uploaded_at": "2026-10-30T14:01:00Z", "review_status": "todo",
 "counts": {"auto_accepted": 2, "needs_review": 1, "accepted": 0, "rejected": 0, "edited": 0}}
```
`review_status` is `todo`, `done` or `empty` (section 1). Every response that contains an Image
(#4, #5, #11, #20) has it.

### 5. GET /projects/{pid}/images?status=needs_review&review_status=todo&offset=0&limit=50
`status` is optional. When given, only images with at least one box in that status are returned.
`review_status` is optional and takes one value. When given, only images with that review status
are returned; an unknown value is 400 `bad_request`. If both filters are given, both must match.
Response `{"images": [Image, ...], "total": 47, "offset": 0, "limit": 50}`.

### 6. GET /images/{iid}/file
Returns the original bytes with the matching `Content-Type`.

### 7. GET /projects/{pid}/classes
Response `{"classes": [Class, ...]}`. **Class**:
```json
{"id": 3, "name": "biax", "description": "a short yellow crossbar hanging from crane cables",
 "min_score": 0.05, "accept_score": null}
```

### 8. PUT /projects/{pid}/classes
The request sends the whole list. Classes are matched by `name`: existing ones are updated, new ones are added, and missing ones are deleted.
```json
{"classes": [
  {"name": "person", "description": "a worker wearing a hard hat, seen from above", "min_score": 0.05, "accept_score": 0.5},
  {"name": "biax",   "description": "a short yellow crossbar hanging from crane cables"}
]}
```
Response `{"classes": [Class, ...]}`. If an omitted class still has annotations, the response is 409 `class_in_use`.

### 9. POST /projects/{pid}/jobs
Request:
```json
{"model": "pipeline", "image_ids": null, "options": {}}
```
`model` is one of the names in `autolabel.models.list_detectors()` (section 3), and
`image_ids: null` means all images. Response 202 is a **Job**:
```json
{"id": 5, "project_id": 1, "kind": "detect", "status": "queued", "progress": 0.0,
 "done": 0, "total": 47, "error": null, "config": {"model": "pipeline", "image_ids": null, "options": {}},
 "created_at": "2026-10-30T14:02:00Z", "started_at": null, "finished_at": null}
```

### 10. GET /jobs/{jid}
Response is a Job. The UI polls it every 1–2 s until `status` is `done` or `failed`.

### 11. GET /images/{iid}/annotations?status=
Response `{"image": Image, "annotations": [Annotation, ...]}`. **Annotation**:
```json
{"id": 101, "image_id": 10, "class_id": 3, "class_name": "biax",
 "bbox": [812.0, 300.5, 240.0, 66.0], "mask": null, "score": 0.42,
 "source": "owlv2_image", "status": "needs_review", "job_id": 5, "reviewer": null,
 "created_at": "2026-10-30T14:03:00Z", "updated_at": "2026-10-30T14:03:00Z"}
```

### 12. POST /images/{iid}/annotations
Request `{"class_id": 1, "bbox": [100, 200, 40, 50], "mask": null, "reviewer": "dhyan"}`.
Response 201 is an Annotation with `source: "human"`, `score: null` and `status: "accepted"`.
If the image's `review_status` is `empty` -> 409 `conflict` (set it back to `todo` with #20 first).

### 13. PATCH /annotations/{aid}
Every field is optional. The status rules from section 1 apply. `status` may be any of the 5
statuses; an unknown one is 400 `bad_request`.
```json
{"status": "accepted", "reviewer": "dhyan"}
{"status": "rejected", "reviewer": "dhyan"}
{"bbox": [810, 298, 250, 70], "class_id": 4, "reviewer": "dhyan"}
{"status": "needs_review", "reviewer": "dhyan"}
{"status": "auto_accepted", "bbox": [810, 298, 250, 70], "class_id": 4, "reviewer": "dhyan"}
```
The response is the updated Annotation. The third example comes back with `status: "edited"`.
Rule: an explicit `status` in the body wins; only when `status` is absent does a bbox / mask /
class change make the box `edited`. So the last example comes back `auto_accepted`. Undo uses
this to put a box back exactly as it was (undo "accept" -> `needs_review`; undo "move" -> the old
bbox and the old status in one call).

### 14. POST /projects/{pid}/export
Request (every field optional):
```json
{"format": "coco", "statuses": ["auto_accepted", "accepted", "edited"], "include_images": false}
```
| Field | Default | Rule |
|---|---|---|
| `format` | `"coco"` | `yolo` or `coco`; anything else -> 400 `bad_request` |
| `statuses` | approved statuses (section 1) | never `rejected`; `rejected`, unknown or empty list -> 400 `bad_request` |
| `include_images` | `false` | `true` puts the image files in the zip too |

Only images with `review_status` `done` or `empty` are exported (section 1, export / push rules);
an `empty` image is exported with no labels. Only boxes whose status is in `statuses` are written.
The response is an `application/zip` download named `<project>_<format>.zip`.
Errors: unknown pid -> 404 `not_found`; bad `format` / `statuses` / body -> 400 `bad_request`;
no `done` or `empty` image in the project -> 409 `conflict`.
Route rule: `app/api/export.py` defines `router = APIRouter()` with the path
`/projects/{pid}/export` (no `/api`); `app/main.py` mounts it under `/api` when the module exists.
The zip layout is defined by the export session in `autolabel/export/`
and documented in its notes. Owner: export.

### 15. (removed)
Replaced by #21. Nothing was built for it.

### 16. GET /models
Response `{"models": [...]}`, which is exactly `autolabel.models.list_detectors()`. Today:
```json
{"models": ["fake"]}
```
The UI uses it to choose the `model` for #9. The default `"pipeline"` in #9 stays; it is
rejected with 400 `unknown_model` until the real pipeline is registered. Owner: backend.

### 17. PATCH /projects/{pid}
Rename. Owner: backend.
```json
{"name": "eot_crane_v2"}
```
Response 200 is a Project (same shape as #1, with `counts`). Errors: empty name or bad body ->
400 `bad_request`; name used by another project -> 409 `name_taken`; unknown pid -> 404
`not_found`. Renaming to its own current name is a no-op 200.

### 18. DELETE /projects/{pid}
Delete a project and everything in it. Owner: backend.
No body. Response **204** with an empty body. Removes the project row and (via ON DELETE CASCADE
plus explicit deletes where needed) its images, classes, jobs and annotations, and the folder
`data/uploads/<pid>/` on disk. If a job of this project is `queued` or `running` -> 409
`conflict` and nothing is deleted. Unknown pid -> 404 `not_found`.

### 19. DELETE /annotations/{aid}
Hard-delete one box. Owner: backend.
No body. Response **204** with an empty body. Allowed for any box, whatever its source or status
(undoing "draw box" or "repeat previous" must remove the boxes it created). Unknown aid -> 404
`not_found`. The UI itself only hard-deletes human boxes (`source: "human"`); for model boxes
"Delete" still sends #13 `{"status": "rejected"}` so they keep counting toward precision.

### 20. PATCH /images/{iid}
Set the image review status. Owner: backend.
```json
{"review_status": "empty", "reviewer": "dhyan"}
{"review_status": "done"}
{"review_status": "todo"}
```
`review_status` is required: `todo`, `done` or `empty`. `reviewer` is optional and is accepted
but not stored for now (the `images` table has no reviewer column). Response 200 is the updated
Image. Any value may be set at any time, except `empty` while the image has a box that is not
`rejected` -> 409 `conflict` and nothing changes. `done` is allowed with boxes still
`needs_review`. Errors: missing or unknown `review_status`, or a bad body -> 400 `bad_request`;
unknown iid -> 404 `not_found`. Setting the current value again is a no-op 200.

### 21. POST /projects/{pid}/roboflow/push
Start a job that uploads images and their boxes to Roboflow. Request body is optional:
```json
{"statuses": ["auto_accepted", "accepted", "edited"]}
```
`statuses` follows the same rules and default as #14. The image / box rules are the same as #14
(section 1). Settings come **only** from `.env` on the server:

| `.env` name | Meaning |
|---|---|
| `ROBOFLOW_API_KEY` | API key. Never in a request, response, job `config`, job `error`, log or DB row. |
| `ROBOFLOW_WORKSPACE` | workspace id, e.g. `vinus-workspace-4ji2t` |
| `ROBOFLOW_PROJECT` | project id, e.g. `eot_crane-yduh1-vhu5p-ohmpg` |

Response 202 is a Job with `kind: "roboflow_push"`, polled with #10. Its `config` is the request
body with defaults filled in (`{"statuses": [...]}`), never the key. `total` = the `done`/`empty`
images whose `roboflow_image_id` is NULL; images already pushed are skipped (so a change made after
a push is not re-sent). A job with `total` 0 finishes `done` at once.
The job uploads one image at a time with the **roboflow Python SDK** (never the Roboflow MCP
connector), with batch name `autolabel_<YYYY-MM-DD>` (job start, UTC), and stores the returned id in
`images.roboflow_image_id` right after each upload, so a failed job can simply be started again.
On an error the job becomes `failed` with a one-line `error` that never contains the key.
Errors (checked in this order): unknown pid -> 404 `not_found`; bad `statuses` / body -> 400
`bad_request`; any of the 3 `.env` names missing or empty -> 409 `conflict` (the message names the
missing variables, never a value); no `done` or `empty` image -> 409 `conflict`.

Who builds what:

| Role | Builds |
|---|---|
| export | `autolabel/export/`: `to_coco`, `to_yolo`, the zip, the image/box selection rules, `roboflow_settings()` (reads the 3 `.env` names, returns the values or the list of missing names) and `push_image(...)` (one SDK upload, returns the Roboflow image id). `app/api/export.py`: routes #14 and #21 (#21 = checks + creates the job row + wakes the worker). Tests `tests/test_export_*.py` with the SDK mocked (no network in tests). |
| backend | B1: add `images.roboflow_image_id` (startup `ALTER TABLE`). B2: the worker runs `roboflow_push` jobs: pick the images, call export's `push_image` per image, save `roboflow_image_id`, update `done`/`progress`, `failed` with a key-free one-line error. B3: a helper the export route can call to insert a job and wake the worker (e.g. `create_job(conn, pid, kind, config, total) -> Job` in `app/db/`). Tests in `tests/test_worker_*.py`. |
| ui | nothing yet (export / push buttons come later, after the endpoints work). |

Export builds against B3's helper name once backend has merged it; until then it may insert the job
row itself in its tests only.

---

## 3. Detector interface (Python)

The models session owns `autolabel/models/` and `autolabel/pipeline.py`. The types below go in
`autolabel/models/base.py`. Everyone else imports them and must not redefine them.

```python
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Protocol
from PIL import Image
from autolabel.evaluate import Box

@dataclass(frozen=True)
class ClassSpec:
    name: str                      # "biax"
    description: str               # text prompt
    min_score: float | None = None

@dataclass(frozen=True)
class Prediction(Box):             # Box gives: image_id: str, category: str, bbox (x,y,w,h px), score
    source: str = ""               # "owlv2", "owlv2_image", "gdino", "pipeline", ...
    mask: dict | None = field(default=None, hash=False, compare=False)   # COCO RLE or None
```
`mask` is left out of hashing and `==` because a dict is not hashable (a frozen dataclass
hashes every field). Two predictions that differ only in their mask compare equal.
```python
class Detector(Protocol):
    name: str                      # the string used in POST .../jobs "model"
    def detect(self, image: Image.Image, image_id: str,
               classes: list[ClassSpec]) -> list[Prediction]: ...

# autolabel/models/__init__.py
def list_detectors() -> list[str]: ...                 # e.g. ["owlv2", "gdino", "pipeline", "fake"]
def get_detector(name: str, **options) -> Detector: ...  # loads weights lazily, once

# autolabel/pipeline.py
def run(images: Iterable[tuple[str, str]],            # (image_id, path on disk)
        classes: list[ClassSpec],
        detector: str = "pipeline",
        options: dict | None = None,
        on_progress: Callable[[int, int], None] | None = None,   # (done, total)
        ) -> Iterator[tuple[str, list[Prediction]]]: ...          # yields one image at a time
```

Rules:
- `category` must be one of the `ClassSpec.name` values passed in, and `image_id` must be
  the string that was passed in. The backend turns these back into `class_id` and `image.id`.
- `score` is between 0 and 1. Detectors drop boxes below `min_score` themselves.
- Because `Prediction` is a `Box`, the existing `evaluate`, `merge.keep_higher` and
  `thresholds` code works on it unchanged.
- The models session also provides a `"fake"` detector. It uses no GPU and returns the same
  2 boxes per image (one per the first two classes, scores 0.9 and 0.3), so the backend can
  test the full job flow on CPU. Until that exists, the backend may use its own fake in its tests.
- The backend's worker calls `pipeline.run` and decides the status of each box. Models code
  never touches the database.

---

## 4. File ownership

| Owner | Owns (creates and edits) | Tests |
|---|---|---|
| backend | `app/api/` (except `export.py`), `app/db/`, `app/worker/`, `app/main.py` | `tests/test_api_*.py`, `tests/test_db_*.py`, `tests/test_worker_*.py` |
| models | `autolabel/models/`, `autolabel/pipeline.py` | `tests/test_models_*.py`, `tests/test_pipeline_*.py` |
| ui | `app/static/` | (manual check against fixtures) |
| export | `autolabel/export/`, `app/api/export.py` | `tests/test_export_*.py` |
| integrator | `CLAUDE.md`, `requirements.txt`, `pyproject.toml`, `README.md`, `docs/CONTRACT.md`, `docs/fixtures/` | merges, runs full `pytest` |

Rules:
- **Don't edit files you don't own.** If you need a change somewhere else, write it in your notes.
- Existing `autolabel/*.py` files (evaluate, owlv2, gdino, merge, coco, ...) are **read-only** for
  everyone. Import them, wrap them, or ask the integrator.
- Each session keeps `docs/agents/<name>.md` up to date with these sections: *Done*, *Packages needed* (with
  versions and VERIFIED/GUESS), *Contract change requests*, and *Open questions*.
- Export works as **pure functions on the API JSON shapes**, so it can be built from the fixtures
  without the database:
  `to_coco(images: list[Image], classes: list[Class], annotations: list[Annotation]) -> dict` and
  `to_yolo(...) -> dict[str, str]` (file name -> file text). `app/api/export.py` only loads rows and
  calls them.
- No session runs anything on the GPU while the paused YOLO training (pid 3863) is holding it.
- The `.env` key is never printed, logged or committed.

---

## Open questions

- `get_detector` caches one instance per **name**. With real models, two jobs with different
  `options` (e.g. another checkpoint) would share the first instance. Key the cache on
  `(name, options)` instead? (models session; no change yet)

---

## 5. Mock data

`docs/fixtures/api_samples.json` holds one example response per endpoint, keyed by `"METHOD path"`.
A value of `null` means 204 with an empty body (#18, #19).
It uses project 1 `eot_crane` with the 4 crane classes, 3 images, and annotations in all 5 statuses
(including one human box and one with a mask).
- **ui:** add a `?mock=1` switch that reads this file instead of calling `/api`. For images, use any
  local gold-set image; the image sizes in the fixtures are 1280x720 (the real crane size).
- **export:** load `images`, `classes` and `annotations` from the fixtures and feed them to `to_coco` and `to_yolo`.
- **backend:** your real responses must match these shapes. A test that compares the keys is a good idea.
