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
  sha256       TEXT,                    -- hex SHA-256 of the uploaded bytes (#4 duplicate check)
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
  kind         TEXT NOT NULL CHECK (kind IN ('detect', 'roboflow_push', 'import')),
  status       TEXT NOT NULL CHECK (status IN ('queued', 'running', 'done', 'failed')),
  progress     REAL NOT NULL DEFAULT 0,   -- 0..1 = done / total
  done         INTEGER NOT NULL DEFAULT 0,
  total        INTEGER NOT NULL DEFAULT 0,
  error        TEXT,                      -- one-line message when status = failed
  config       TEXT NOT NULL,             -- JSON: the request body of POST .../jobs
  result       TEXT,                      -- JSON summary; set by import jobs (#22), NULL otherwise
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
  source      TEXT NOT NULL,              -- 'owlv2' | 'owlv2_image' | 'gdino' | 'pipeline' | 'human' | 'import'
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
| Box imported by #22 | `needs_review` (source `import`, `score` NULL, `job_id` = the import job) |
| Reviewer approves | `accepted` |
| Reviewer deletes | `rejected` (the row is kept so we can measure precision later) |
| Reviewer changes box, mask or class | `edited` |
| Reviewer undoes a change | the status sent in the request |

An explicit `status` in a PATCH (#13) always wins. Only when `status` is absent does a
bbox / mask / class change make the box `edited`.

A new `detect` job on an image deletes that image's earlier **model** boxes that are
still `auto_accepted` or `needs_review`. It never touches human-reviewed rows.
A **model** box is one whose `source` is not `human` and not `import`, so imported boxes are
never deleted by a detect job, even while `needs_review`.

Overlap rule (#9): after that delete, a new model box is dropped (not stored) when the image
still has a box of the **same class** whose status is not `rejected` and whose IoU with the new
box is **>= 0.5** (`autolabel.evaluate.iou`). This keeps a detect job from duplicating imported,
human or already reviewed boxes. The check is done by `pipeline.drop_overlapping` (section 3).

`jobs.result` is the JSON summary of an import job (#22) and NULL for other kinds. On startup
`migrate()` adds it (`ALTER TABLE jobs ADD COLUMN result TEXT`) if missing. SQLite cannot change a
CHECK, so if the `jobs` table's CHECK does not allow `'import'`, `migrate()` rebuilds the table
(create `jobs_new` with the schema above, copy every row, drop `jobs`, rename, inside one
transaction with foreign keys off), keeping all ids.

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

`images.sha256` is the hex SHA-256 of the file bytes, set by #4. On startup `migrate()` adds the
column (`ALTER TABLE images ADD COLUMN sha256 TEXT`) if missing, then fills it for every row where
it is NULL by hashing the file at `path` (a missing file leaves it NULL). It is not part of the Image JSON.

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
| 22 | POST | `/projects/{pid}/import` | start bulk import job (zip or many image files) |

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
{"images": [Image, ...],
 "skipped": [{"file_name": "a.jpg", "reason": "duplicate"}],
 "renamed": [{"from": "frame_0001.jpg", "to": "frame_0001_2.jpg"}]}
```
Duplicate rules (by content, not name), checked file by file in upload order:

| Upload | Result |
|---|---|
| Same bytes (sha256) as an image already in the project, or earlier in this request, whatever the name | skipped, `reason: "duplicate"` |
| Same name as an existing image, different bytes | saved as `stem_2.ext` (`stem_3.ext`, ... if taken); the Image has the new `file_name`; listed in `renamed` |
| New name, new bytes | saved as now |
| Not a jpg/png | skipped, `reason: "not an image"` |

`skipped` and `renamed` are always present (empty lists when nothing applies).
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
 "result": null, "created_at": "2026-10-30T14:02:00Z", "started_at": null, "finished_at": null}
```
On images that already have boxes, the overlap rule from section 1 applies: a model box is dropped
when an existing non-rejected box of the same class overlaps it at IoU >= 0.5. The worker first
deletes the earlier model boxes (section 1), then passes the image's remaining non-rejected boxes to
`pipeline.drop_overlapping` and stores what it returns.

### 10. GET /jobs/{jid}
Response is a Job. The UI polls it every 1–2 s until `status` is `done` or `failed`.
Every Job has `result`: `null` for `detect` and `roboflow_push`, the import summary for `import`
(#22).

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
Errors: unknown pid -> 404 `not_found`; `include_images: true` and an exported image's file is
missing on disk -> 404 `not_found` (no zip is sent); bad `format` / `statuses` / body -> 400
`bad_request`; no `done` or `empty` image in the project -> 409 `conflict`.
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
| ui | Projects page buttons (Export, Push to Roboflow). |

Export builds against B3's helper name once backend has merged it; until then it may insert the job
row itself in its tests only.

Push interface (owner: export; importable as `from autolabel.export import push_image, roboflow_settings`):
```python
def roboflow_settings() -> object | list[str]: ...
    # the settings (e.g. a dict) when all 3 .env names are set, else the LIST of missing names
def push_image(settings, image: dict, boxes: list[dict], class_names: list[str],
               batch_name: str) -> str: ...
    # one SDK upload; returns the Roboflow image id
    # image       = {"id", "file_name", "path", "width", "height"}
    # boxes       = [{"class_name": "biax", "bbox": [x, y, w, h]}, ...]   (pixels, section conventions)
    # class_names = project class order (#7)
```
The **worker** (backend) owns image selection (`done`/`empty`, `roboflow_image_id` NULL), skipping,
saving the returned id, progress and errors. It passes `push_image` only boxes whose status is in the
job's `statuses` (#14's default approved statuses when the body gave none; never `rejected`), and
`boxes = []` for an `empty` image. `push_image` does no filtering and never touches the database.
If `settings` is a dict, values under a key whose name contains "key" are scrubbed from job errors.

### 22. POST /projects/{pid}/import
Start a job that adds many images (and their existing labels) to a project.
The request is `multipart/form-data` with one or more `files` fields, either:

| Upload | Meaning |
|---|---|
| exactly one `.zip` part, nothing else | zip import; the format is detected inside the zip (below) |
| one or more non-zip parts (folder upload) | images only; the part file name may carry the folder path (e.g. `site_a/f_001.jpg`) |

Response 202 is a Job with `kind: "import"`, polled with #10. Its `config` is
`{"upload": "zip", "file_name": "crane_yolo.zip"}` or `{"upload": "files", "count": 312}`;
`total` = number of image files (`.jpg`/`.jpeg`/`.png` names) in the zip or in the parts.

Formats inside a zip (detected in this order, first match wins):

| Format | Detected when | Boxes come from |
|---|---|---|
| `coco` | a `.json` file with `images`, `annotations` and `categories` keys | `annotations[].bbox` (already `[x, y, w, h]` px). If there are several such files (Roboflow COCO exports have one per split folder), each applies to the images in its own folder and below. Images are matched by base name of `images[].file_name`. |
| `yolo` | a `labels/` folder with `.txt` files next to an `images/` folder, plus a `data.yaml` (`names`: list or `{index: name}`) or a `classes.txt` (one name per line); the top-most one is used. Roboflow YOLO exports (`train/`, `valid/`, `test/` each with `images/` + `labels/`, `data.yaml` at the root) match. | `labels/<stem>.txt` for `images/<stem>.<ext>` in the same folder; each line `class cx cy w h`, normalized 0..1, converted to `[x, y, w, h]` px with the image size |
| `images` | anything else | none |

Rules:

| Rule | Value |
|---|---|
| Duplicates | same rule as #4, image by image in zip order: same sha256 as an image in the project or earlier in this job -> not added, listed in `duplicates`, and its boxes are **not** imported; same name + new bytes -> `stem_2.ext`, listed in `renamed`. `file_name` = the base name (folders are dropped). |
| Imported boxes | status `needs_review`, source `import`, `score` NULL (a COCO `score` is ignored), `mask` NULL (COCO segmentation is ignored), `job_id` = the import job, `reviewer` NULL. Boxes are clipped to the image; a box with no area left is skipped. |
| Imported images | `review_status` `todo` |
| Classes | matched to the project's classes by **exact name** (case-sensitive). Names with at least one box in the upload that the project lacks are added at the **end** of the class list (so existing YOLO export indexes do not move), in file order (COCO `categories` order, YOLO `names` order), with `description` = the name and `min_score` / `accept_score` NULL. Categories with no boxes are not created (e.g. Roboflow's COCO super-category). |
| Bad or unknown files | never fail the job: skipped and listed in `skipped` with a reason. Non-image files are skipped (`"not an image"`), except the label files used by the detected format. A bad label line, an unknown class index / `category_id`, a label file or COCO image with no matching image, or a file that will not open as an image are skipped too. An image with no label file is imported with no boxes. |
| Safety (checked by the route, before the job exists) | a zip entry or part file name with `..` as a path part, or an absolute path (`/x`, `\x`, `C:\x`) -> 400 `bad_request`. Total unpacked size (sum of the zip's uncompressed sizes, or of the parts) above the cap -> 400 `bad_request`. Cap: `AUTOLABEL_IMPORT_MAX_GB` env var, default **5** (GB = 10^9 bytes). While unpacking, a file that grows past its declared size or past the cap is stopped and skipped (`"too large"`). |
| Upload storage | the upload is saved under `data/imports/<job_id>/` and that folder is deleted when the job ends (`done` or `failed`). |
| `failed` | only for an unexpected error; images and boxes already added stay. `result` may then be `null`. |

`result` of a `done` import job:
```json
{"format": "yolo",
 "images_added": 118,
 "duplicates": ["valid/images/f_0007.jpg"],
 "renamed": [{"from": "f_0001.jpg", "to": "f_0001_2.jpg"}],
 "boxes_by_class": {"person": 240, "biax": 31},
 "classes_created": ["biax"],
 "skipped": [{"file_name": "README.roboflow.txt", "reason": "not an image"},
             {"file_name": "train/labels/f_0042.txt", "reason": "line 3: unknown class index 7"}]}
```
`format` is `coco`, `yolo` or `images` (always `images` for a folder upload). `duplicates`, `renamed`,
`classes_created` and `skipped` are lists (empty when nothing applies); `boxes_by_class` counts only
boxes actually stored (by class name; `{}` when none). Paths in `duplicates` / `skipped` are relative
to the zip root or the folder.

Errors (checked in this order): unknown pid -> 404 `not_found`; no `files` part, a `.zip` sent
together with other parts, more than one `.zip`, a zip that will not open, an unsafe path or an
upload above the cap -> 400 `bad_request`.

Who builds what (#22 and the #9 change):

| Role | Builds |
|---|---|
| export | `autolabel/importers/` (pure functions, no database): the interface below + `tests/test_importers_*.py` (small zips/folders built in `tmp_path`). |
| backend | Route #22 in `app/api/` (checks via `check_zip` / `check_names`, saves the upload, `create_job(..., kind="import")`); the worker runs `import` jobs (extract, `plan_import`, #4 duplicate rule shared with #4, add classes, insert images + boxes, `done`/`progress` per image, `result`, delete the temp folder); `jobs.result` + the `jobs` table rebuild in `migrate()`; `result` in every Job JSON; the #9 worker change (delete old model boxes, then `drop_overlapping`). Tests in `tests/test_api_*.py` / `tests/test_worker_*.py`. |
| models | `drop_overlapping` in `autolabel/pipeline.py` (section 3) + `tests/test_pipeline_*.py`. |
| ui | A drop zone for one `.zip`, plus a folder picker (`<input webkitdirectory>`, part file name = `webkitRelativePath`) that sends the folder's files to #22; polls #10; shows `result` as a summary (images added, duplicates, renamed, boxes per class, classes created, skipped files with reason). |

Import interface (owner: export; `from autolabel.importers import ...`):
```python
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

class ImportRejected(ValueError): ...   # message is safe to show in a 400

def check_names(names: list[str]) -> None: ...
    # raises ImportRejected if any name has '..' as a path part or is absolute
def check_zip(zip_path: Path, max_bytes: int) -> int: ...
    # raises ImportRejected: not a zip, unsafe entry name (check_names), sum of uncompressed
    # sizes > max_bytes. Returns the number of image entries (the job's total).
def extract_zip(zip_path: Path, dest: Path, max_bytes: int) -> list[dict]: ...
    # unpacks under dest only; returns skipped [{"file_name", "reason"}] (e.g. "too large")

@dataclass(frozen=True)
class ImportItem:
    rel_path: str            # path under the root, '/' separators
    boxes: list[dict]        # [{"class_name": "biax", "bbox": [x, y, w, h]}], px, clipped

@dataclass
class ImportPlan:
    format: str              # "coco" | "yolo" | "images"
    items: list[ImportItem]  # one per image file, in path order
    class_names: list[str]   # names with >= 1 box, in file order
    skipped: list[dict]      # [{"file_name", "reason"}]

def plan_import(root: Path, images_only: bool = False) -> ImportPlan: ...
    # reads an unpacked zip (or saved folder upload with images_only=True); opens images
    # only to read their size. Never touches the database.
```

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

def drop_overlapping(predictions: list[Prediction],
                     existing: list[Box],              # the image's non-rejected boxes, category = class name
                     iou_threshold: float = 0.5,
                     ) -> list[Prediction]: ...
    # drops a prediction when an existing box with the same category has
    # evaluate.iou(pred.bbox, box.bbox) >= iou_threshold; keeps the order of the rest
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
| export | `autolabel/export/`, `autolabel/importers/`, `app/api/export.py` | `tests/test_export_*.py`, `tests/test_importers_*.py` |
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
- #22 folder upload is images only (as requested). Run the zip format detection on folder uploads
  too, so a dropped YOLO / COCO folder keeps its labels? (`plan_import(images_only=False)`; user to decide)

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
