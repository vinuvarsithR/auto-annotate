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
| Polygons | `[[x, y], ...]` normalized 0..1, top-left origin, at least 3 points, one outer ring, or `null` |
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
  result       TEXT,                      -- JSON summary; set by import jobs (#22) and examples-mode detect jobs, NULL otherwise
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
  polygon     TEXT,                       -- JSON [[x, y], ...] normalized 0..1, or NULL (plain box)
  score       REAL,                       -- NULL for human-drawn boxes
  source      TEXT NOT NULL,              -- 'owlv2' | 'owlv2_image' | 'yoloe_examples' | 'gdino' | 'pipeline' | 'human' | 'import'
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
| Reviewer changes box, mask, polygon or class | `edited` |
| Reviewer undoes a change | the status sent in the request |

An explicit `status` in a PATCH (#13) always wins. Only when `status` is absent does a
bbox / mask / polygon / class change make the box `edited`.

A new `detect` job on an image deletes that image's earlier **model** boxes that are
still `auto_accepted` or `needs_review`. It never touches human-reviewed rows.
A **model** box is one whose `source` is not `human` and not `import`, so imported boxes are
never deleted by a detect job, even while `needs_review`.
A class skipped by an examples-mode job (#9) did not run, so its earlier model boxes are kept.

Overlap rule (#9): after that delete, a new model box is dropped (not stored) when the image
still has a box of the **same class** whose status is not `rejected` and whose IoU with the new
box is **>= 0.5** (`autolabel.evaluate.iou`). This keeps a detect job from duplicating imported,
human or already reviewed boxes. The check is done by `pipeline.drop_overlapping` (section 3).

`jobs.result` is the JSON summary of an import job (#22) or an examples-mode detect job (#9), and
NULL for other jobs. On startup
`migrate()` adds it (`ALTER TABLE jobs ADD COLUMN result TEXT`) if missing. SQLite cannot change a
CHECK, so if the `jobs` table's CHECK does not allow `'import'`, `migrate()` rebuilds the table
(create `jobs_new` with the schema above, copy every row, drop `jobs`, rename, inside one
transaction with foreign keys off), keeping all ids.

`annotations.polygon` is optional. x, y, w, h stay NOT NULL on every row. Rule: when `polygon` is
set, the bbox is **always** the polygon's bounding box, computed by the server (min/max of the
points times the image width/height). `migrate()` adds the column on startup
(`ALTER TABLE annotations ADD COLUMN polygon TEXT`) if missing; old rows stay NULL (plain boxes), no
rebuild is needed. `drop_overlapping` uses the bbox only. Model boxes have `polygon` NULL, except
`yoloe` boxes (#9, examples mode). `polygon` and `mask` are independent; #14 and #21 use `polygon`
first (see #14).

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
| 400 | `unknown_mode` | #9 `mode` not in the detector's #16 list |
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
{"model": "pipeline", "mode": null, "image_ids": null, "options": {}}
```
`model` is one of the names in `autolabel.models.list_detectors()` (section 3), and
`image_ids: null` means all images.

| `mode` in the request | Result |
|---|---|
| absent or `null` | the detector's first mode in #16, or `null` if it has none. Existing callers keep today's behaviour. |
| a mode in the detector's #16 list | used |
| anything else | 400 `unknown_mode` |

`mode` is a top-level field, never in `options`: `get_detector` builds one instance per name and uses
`options` only the first time. `config.mode` stores the mode that was resolved. Jobs created before
this change have no `config.mode`.

Response 202 is a **Job**:
```json
{"id": 5, "project_id": 1, "kind": "detect", "status": "queued", "progress": 0.0,
 "done": 0, "total": 47, "error": null, "config": {"model": "pipeline", "mode": null, "image_ids": null, "options": {}},
 "result": null, "created_at": "2026-10-30T14:02:00Z", "started_at": null, "finished_at": null}
```
On images that already have boxes, the overlap rule from section 1 applies: a model box is dropped
when an existing non-rejected box of the same class overlaps it at IoU >= 0.5. The worker first
deletes the earlier model boxes (section 1), then passes the image's remaining non-rejected boxes to
`pipeline.drop_overlapping` and stores what it returns. This is the same in every mode.

#### Examples mode
A detector learns from the project's own boxes, with no training. Detectors and modes:

| `model` | `mode` | Status | Stored `source` | Polygon |
|---|---|---|---|---|
| `yoloe` | `examples` | new | `yoloe_examples` | yes |
| `owlv2` | `text` | exists, unchanged | `owlv2` | no |
| `owlv2` | `examples` | new | `owlv2_image` | no |
| `fake` | `text`, `examples` | `examples` returns its usual boxes for classes with examples | `fake` | no |

Which boxes are examples:

| Box status | Example? |
|---|---|
| `accepted` or `edited`, any `source` (`human`, `import` or a model) | yes |
| `rejected`, `auto_accepted` or `needs_review`, any `source` | no |

So a reviewed model box teaches the next run; a box nobody has reviewed does not.

| Rule | Value |
|---|---|
| Scope | all images of the project, whatever `image_ids` says |
| What is used | the bbox only; a `polygon` or `mask` on the row is ignored |
| Clean-up (worker) | bbox clipped to the image; no area left, or image file missing -> not an example |
| Cap | at most **30** examples per class, evenly spaced through the class's rows in annotation id order. Fixed value, not an option. |
| Storage | embeddings are computed at job start and kept in memory for that job. Nothing new is stored. |
| Class with no examples | not run, no error. Listed in `result.skipped_classes`; its earlier model boxes are kept (section 1). |
| No class left | the job finishes `done` at once with `done` = `total`; no images are run and nothing is deleted. |
| Unreadable example image | the job becomes `failed`, like an unreadable target image |

Worker order: (1) read and clean the examples; (2) call `pipeline.prepare_examples` unless no class
has examples (`status` is `running`, `progress` stays 0 while it embeds); (3) write `result` to the
job; (4) run the images as above, deleting earlier model boxes of the classes that ran only.

`result` of an examples-mode job, readable from step 3 on:
```json
{"mode": "examples",
 "examples_by_class": {"person": 30, "biax": 6},
 "skipped_classes": [{"class_name": "sb", "reason": "no examples"},
                     {"class_name": "rb", "reason": "no usable examples"}]}
```
`examples_by_class` counts the examples actually used, and only classes that ran. `reason` is
`no examples` (nothing left after the worker's clean-up) or `no usable examples` (the detector
rejected all of them). `skipped_classes` is `[]` when nothing was skipped.

Scores: `score` is always 0..1, so `min_score` and `accept_score` work the same way in every mode.
They are one setting per class, shared by all modes; a value tuned for one mode is not tuned for another.

| Detector | Prototype | Score |
|---|---|---|
| `owlv2` | per class, the mean of `query_embedding` over its examples, one query per class | `sigmoid(class_predictor logit)`, as `detect_image_guided` does today. Never `post_process_image_guided_detection`: it rescales per image, so the best box becomes 1.0 even with no object. |
| `yoloe` | per class, the mean of its example embeddings, L2-normalized, given to `set_classes(names, vpe)` | `boxes.conf` as returned; the head already applies the sigmoid. Pass `conf` = the lowest class floor, because the ultralytics default would hide low scores. |

Default floor when `min_score` is NULL: 0.05, as `owlv2` and `yolo`.

`yoloe` rules:

| Rule | Value |
|---|---|
| Weights | `AUTOLABEL_YOLOE_WEIGHTS`, else `models/yoloe-26s-seg.pt` under the repo root (gitignored). File missing -> the job fails with a one-line error; it never downloads at job time. Must be a `-seg` file, not `-seg-pf`. |
| Enabling | `AUTOLABEL_DETECTORS` contains `yoloe`, as the other real detectors |
| Image size | 640, for example embedding and for detection |
| Model state | `set_classes` changes the model, so each job loads a fresh model in `prepare_examples` and the instance from `get_detector` keeps no job state |
| Example too small | prompts are rasterized on a 1/8 grid, and an empty prompt silently becomes a whole-image embedding. A box whose shorter side is under 8 px after scaling to 640 is not used. |
| Mask to polygon | predict with `retina_masks=True`; `cv2.findContours` with `RETR_EXTERNAL`; keep the **largest part** by area (holes and other parts are dropped); simplify with `cv2.approxPolyDP`, closed, tolerance **0.001 x the image's longest side** (1.28 px at 1280); divide by width / height, round to 6 decimals |
| Bbox | the polygon's bounding box, computed by the detector after simplifying (section 1 bbox rule) |
| Bad polygon | fewer than 3 points or zero area -> `polygon` None, the model's own bbox is kept. `mask` is always None. |

### 10. GET /jobs/{jid}
Response is a Job. The UI polls it every 1–2 s until `status` is `done` or `failed`.
Every Job has `result`: the import summary for `import` (#22), the examples summary for an
examples-mode `detect` job (#9), and `null` otherwise. It stays `null` until #9 step 3.

### 11. GET /images/{iid}/annotations?status=
Response `{"image": Image, "annotations": [Annotation, ...]}`. **Annotation**:
```json
{"id": 101, "image_id": 10, "class_id": 3, "class_name": "biax",
 "bbox": [812.0, 300.5, 240.0, 66.0], "mask": null, "polygon": null, "score": 0.42,
 "source": "owlv2_image", "status": "needs_review", "job_id": 5, "reviewer": null,
 "created_at": "2026-10-30T14:03:00Z", "updated_at": "2026-10-30T14:03:00Z"}
```
`polygon` is always present: `null` for a box, else the point list (section conventions).

### 12. POST /images/{iid}/annotations
Request `{"class_id": 1, "bbox": [100, 200, 40, 50], "mask": null, "reviewer": "dhyan"}`.
Optional `polygon`: with it, `bbox` may be left out; if both are sent, `bbox` is ignored and
computed from the polygon. Without `polygon`, `bbox` is required, as before.
```json
{"class_id": 1, "polygon": [[0.1, 0.2], [0.3, 0.2], [0.2, 0.4]], "reviewer": "dhyan"}
```
A bad `polygon` (fewer than 3 points, a point that is not 2 finite numbers in 0..1, or zero area)
-> 400 `bad_request`. The server never clips an API polygon; only #22 does.
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
`polygon` follows the same validation as #12:

| Body | Effect |
|---|---|
| `polygon: [[x, y], ...]` | replaces the polygon, or turns a box into a polygon; bbox recomputed. A `bbox` in the same body is ignored. |
| `polygon: null` | removes the polygon; a `bbox` in the same body is applied, otherwise the bbox stays as it is |
| `bbox` only, on a row with a polygon | the polygon is scaled and moved to fill the new bbox, so box-only clients keep working and the bbox rule still holds. Points are clamped to 0..1, then the bbox is recomputed. |
| `polygon` absent | polygon untouched |

A bbox-only PATCH on a polygon row whose new bbox leaves no area inside the image -> 400 `bad_request`, nothing changes.

The response is the updated Annotation. The third example comes back with `status: "edited"`.
Rule: an explicit `status` in the body wins; only when `status` is absent does a bbox / mask /
polygon / class change make the box `edited`. So the last example comes back `auto_accepted`. Undo uses
this to put a box back exactly as it was (undo "accept" -> `needs_review`; undo "move" -> the old
bbox and the old status in one call).

### 14. POST /projects/{pid}/export
Request (every field optional):
```json
{"format": "coco", "statuses": ["auto_accepted", "accepted", "edited"], "include_images": false}
```
| Field | Default | Rule |
|---|---|---|
| `format` | `"coco"` | `yolo`, `yolo-seg` or `coco`; anything else -> 400 `bad_request` |
| `statuses` | approved statuses (section 1) | never `rejected`; `rejected`, unknown or empty list -> 400 `bad_request` |
| `include_images` | `false` | `true` puts the image files in the zip too |

Only images with `review_status` `done` or `empty` are exported (section 1, export / push rules);
an `empty` image is exported with no labels. Only boxes whose status is in `statuses` are written.
The response is an `application/zip` download named `<project>_<format>.zip`.

| Format | Row with polygon | Row without polygon |
|---|---|---|
| `yolo` | `class cx cy w h` of its bbox | `class cx cy w h` |
| `yolo-seg` | `class x1 y1 x2 y2 ...`, normalized | same line shape with the 4 corners of the bbox, clipped to the image; no area left -> left out |
| `coco` | `bbox` + `segmentation: [[x1, y1, x2, y2, ...]]` in **pixels**; `area` = polygon area | `bbox`; `segmentation` = `mask` if set, else `[]`; `area` = w * h |

`yolo-seg` has the same zip layout as `yolo` (`data.yaml`, `classes.txt`, `labels/`, `images/`).
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
Response `{"models": [...], "modes": {...}}`. `models` is exactly
`autolabel.models.list_detectors()`; `modes` has one key per model, from
`autolabel.models.detector_modes(name)`, default mode first. `[]` = the detector has no modes. Today:
```json
{"models": ["fake"], "modes": {"fake": ["text", "examples"]}}
```
With `AUTOLABEL_DETECTORS=owlv2,yolo,yoloe`:
```json
{"models": ["fake", "owlv2", "yolo", "yoloe"],
 "modes": {"fake": ["text", "examples"], "owlv2": ["text", "examples"], "yolo": [], "yoloe": ["examples"]}}
```
The UI uses it to choose the `model` and `mode` for #9. The default `"pipeline"` in #9 stays; it is
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
Start a job that uploads images and their boxes to Roboflow. Polygons are sent when present (see
the push interface). Request body is optional:
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
    # boxes       = [{"class_name": "biax", "bbox": [x, y, w, h], "polygon": [[x, y], ...] | None}, ...]
    #               bbox in pixels, polygon normalized (section conventions); polygon may be absent
    # class_names = project class order (#7)
```
The **worker** (backend) owns image selection (`done`/`empty`, `roboflow_image_id` NULL), skipping,
saving the returned id, progress and errors. It passes `push_image` only boxes whose status is in the
job's `statuses` (#14's default approved statuses when the body gave none; never `rejected`), and
`boxes = []` for an `empty` image. `push_image` does no filtering and never touches the database.
The upload format stays one-image COCO through `project.single_upload`. `coco_for_image` writes
`segmentation` and `area` as in #14 for rows with a polygon; rows without one keep `segmentation: []`.

Push changes by role:

| Code | Owner | Change |
|---|---|---|
| `coco_for_image` in `autolabel/export/roboflow_push.py` | export | `segmentation` + `area` for rows with a polygon; the 4-corner fallback below if it is needed |
| box list in `app/worker/worker.py:286` | backend | add `"polygon"` to each box passed to `push` |

Polygon test upload (approved 2026-11-04, **export phase only**): with the roboflow Python SDK and the
key from `.env`, upload to project `test-g9qbh` and no other project (`ROBOFLOW_PROJECT` is not used
for this test). Upload a mixed image (one polygon row, one box row with `segmentation: []`) and read
back what Roboflow stored. If the box row was dropped, `coco_for_image` sends rows without a polygon
as 4-corner polygons (the bbox corners, clipped to the image, as in `yolo-seg`) on every image. Export
records the result in `docs/agents/export.md`, with the Roboflow project type it saw.
If `settings` is a dict, values under a key whose name contains "key" are scrubbed from job errors.

### 22. POST /projects/{pid}/import
Start a job that adds many images (and their existing labels) to a project.
The request is `multipart/form-data` with one or more `files` fields, either:

| Upload | Meaning |
|---|---|
| exactly one `.zip` part, nothing else | zip import; the format is detected inside the zip (below) |
| any other set of parts (folder upload) | each part's file name is the file's **relative path** in the picked folder (the UI sends `webkitRelativePath`, e.g. `crane/train/labels/f_001.txt`). The server saves the parts at those paths and treats the set **exactly like the contents of a zip**: same format detection, same rules, same path safety checks. A `.zip` part in a folder upload is just a non-image file (skipped, `"not an image"`). |

Response 202 is a Job with `kind: "import"`, polled with #10. Its `config` is
`{"upload": "zip", "file_name": "crane_yolo.zip"}` or `{"upload": "files", "count": 312}`
(`count` = number of parts); `total` = number of image files (`.jpg`/`.jpeg`/`.png` names) in the
zip or in the parts.

Formats inside a zip or a folder upload (below, "the upload"; detected in this order, first match wins):

| Format | Detected when | Boxes come from |
|---|---|---|
| `coco` | a `.json` file with `images`, `annotations` and `categories` keys | `annotations[].bbox` (already `[x, y, w, h]` px) and `annotations[].segmentation` (table below). If there are several such files (Roboflow COCO exports have one per split folder), each applies to the images in its own folder and below. Images are matched by base name of `images[].file_name`. |
| `yolo` | a `labels/` folder with `.txt` files next to an `images/` folder, plus a `data.yaml` (`names`: list or `{index: name}`) or a `classes.txt` (one name per line); the top-most one is used. Roboflow YOLO exports (`train/`, `valid/`, `test/` each with `images/` + `labels/`, `data.yaml` at the root) match. | `labels/<stem>.txt` for `images/<stem>.<ext>` in the same folder; each line is one row by value count (table below), normalized 0..1 |
| `images` | anything else | none |

YOLO rows (per row; mixed rows in one file are allowed):

| Values on the line | Result |
|---|---|
| 5 | box `class cx cy w h`, converted to `[x, y, w, h]` px with the image size |
| odd, >= 7 | polygon `class x1 y1 x2 y2 ...` (at least 3 points); bbox computed |
| anything else | line skipped: `"line 3: expected 5 values or an odd count >= 7, got 6"` |

COCO `segmentation`, per annotation:

| Value | Result |
|---|---|
| missing or `[]` | box |
| list with one polygon part (flat `[x1, y1, x2, y2, ...]` px, >= 3 points) | polygon (divided by image width/height); bbox computed |
| list with more than one part | box from `bbox`, `polygon` NULL, counted in the note below |
| RLE (a dict) | box from `bbox`, `polygon` NULL, counted in the note below. RLE is not decoded. |
| anything else, or a polygon with zero area | box from `bbox`, `polygon` NULL, counted in the note below |

Note: one `skipped` entry per COCO file, not one per annotation, e.g.
`{"file_name": "train/_annotations.coco.json", "reason": "12 segmentations imported as box only (RLE, multi-part or unusable polygon)"}`.
The box is stored, so these rows still count in `boxes_by_class`.

Rules:

| Rule | Value |
|---|---|
| Duplicates | same rule as #4, image by image in path order: same sha256 as an image in the project or earlier in this job -> not added, listed in `duplicates`, and its boxes are **not** imported; same name + new bytes -> `stem_2.ext`, listed in `renamed`. `file_name` = the base name (folders are dropped). |
| Imported boxes | status `needs_review`, source `import`, `score` NULL (a COCO `score` is ignored), `mask` NULL, `job_id` = the import job, `reviewer` NULL. Boxes are clipped to the image; a box with no area left is skipped. Polygon points are clamped to 0..1; a YOLO polygon with no area left is skipped like a bad line. |
| Imported images | `review_status` `todo` |
| Classes | matched to the project's classes by **exact name** (case-sensitive). Names with at least one box in the upload that the project lacks are added at the **end** of the class list (so existing YOLO export indexes do not move), in file order (COCO `categories` order, YOLO `names` order), with `description` = the name and `min_score` / `accept_score` NULL. Categories with no boxes are not created (e.g. Roboflow's COCO super-category). |
| Bad or unknown files | never fail the job: skipped and listed in `skipped` with a reason. Non-image files are skipped (`"not an image"`), except the label files used by the detected format. A bad label line, an unknown class index / `category_id`, a label file or COCO image with no matching image, or a file that will not open as an image are skipped too. An image with no label file is imported with no boxes. |
| Safety (checked by the route, before the job exists) | a zip entry name or a part file name (relative path) with `..` as a path part, or an absolute path (`/x`, `\x`, `C:\x`) -> 400 `bad_request`; two parts with the same relative path -> 400 `bad_request`. Total unpacked size (sum of the zip's uncompressed sizes, or of the parts) above the cap -> 400 `bad_request`. Cap: `AUTOLABEL_IMPORT_MAX_GB` env var, default **5** (GB = 10^9 bytes). While unpacking, a file that grows past its declared size or past the cap is stopped and skipped (`"too large"`). |
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
`format` is `coco`, `yolo` or `images`, for a zip and a folder upload alike. `duplicates`, `renamed`,
`classes_created` and `skipped` are lists (empty when nothing applies); `boxes_by_class` counts only
boxes actually stored (by class name; `{}` when none). Paths in `duplicates` / `skipped` are relative
to the zip root or the folder.

Errors (checked in this order): unknown pid -> 404 `not_found`; no `files` part, a single `.zip`
that will not open, an unsafe path, a repeated path (for a zip, only entries with the same path but
different size or CRC), or an upload above the cap -> 400 `bad_request`.

Who builds what (#22 and the #9 change):

| Role | Builds |
|---|---|
| export | `autolabel/importers/` (pure functions, no database): the interface below + `tests/test_importers_*.py` (small zips/folders built in `tmp_path`). |
| backend | Route #22 in `app/api/` (checks via `check_zip` / `check_names`, saves the zip, or the folder parts at their relative paths under `data/imports/<job_id>/files/`, `create_job(..., kind="import")`); the worker runs `import` jobs (extract a zip, then `plan_import` on the folder in both cases, #4 duplicate rule shared with #4, add classes, insert images + boxes, `done`/`progress` per image, `result`, delete the temp folder); `jobs.result` + the `jobs` table rebuild in `migrate()`; `result` in every Job JSON; the #9 worker change (delete old model boxes, then `drop_overlapping`). Tests in `tests/test_api_*.py` / `tests/test_worker_*.py`. |
| models | `drop_overlapping` in `autolabel/pipeline.py` (section 3) + `tests/test_pipeline_*.py`. |
| ui | A drop zone for one `.zip`, plus a folder picker (`<input webkitdirectory>`) that sends **every** file of the folder to #22 with part file name = `webkitRelativePath` (`formData.append("files", file, file.webkitRelativePath)`), labels included; polls #10; shows `result` as a summary (images added, duplicates, renamed, boxes per class, classes created, skipped files with reason). |

Import interface (owner: export; `from autolabel.importers import ...`):
```python
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

class ImportRejected(ValueError): ...   # message is safe to show in a 400

def check_names(names: list[str]) -> None: ...
    # raises ImportRejected if any name has '..' as a path part, is absolute, or appears twice
    # (used for folder-upload and loose-file names only)
def check_zip(zip_path: Path, max_bytes: int) -> int: ...
    # raises ImportRejected: not a zip, unsafe entry name ('..' path part or absolute), sum of
    # uncompressed sizes > max_bytes. Returns the number of image entries (the job's total).
def extract_zip(zip_path: Path, dest: Path, max_bytes: int) -> list[dict]: ...
    # unpacks under dest only; returns skipped [{"file_name", "reason"}] (e.g. "too large")
    # check_zip and extract_zip: entries with the same path, size and CRC count as one: the first
    # is kept, the rest are ignored and do not count toward the size cap. Same path with a
    # different size or CRC raises ImportRejected 'path appears twice: NAME'. Compared from
    # ZipInfo only.

@dataclass(frozen=True)
class ImportItem:
    rel_path: str            # path under the root, '/' separators
    boxes: list[dict]        # [{"class_name": "biax", "bbox": [x, y, w, h], "polygon": [[x, y], ...] | None}]
                             # bbox px, clipped; polygon normalized, clamped; bbox = polygon's bbox when set

@dataclass
class ImportPlan:
    format: str              # "coco" | "yolo" | "images"
    items: list[ImportItem]  # one per image file, in path order
    class_names: list[str]   # names with >= 1 box, in file order
    skipped: list[dict]      # [{"file_name", "reason"}]

def plan_import(root: Path) -> ImportPlan: ...
    # reads an unpacked zip or a saved folder upload (same rules for both); opens images
    # only to read their size. Never touches the database.
```

---

### Examples mode: who builds what

| Role | Builds |
|---|---|
| models | `autolabel/models/yoloe.py` (weights, `prepare_examples`, `mask_to_polygon` as a pure function); `prepare_examples` on `owlv2` and `fake`; `Example`, `ExampleRun`, `Prediction.polygon` in `base.py`; `detector_modes` and the `yoloe` registry entry; `pipeline.prepare_examples` and the `example_run` argument of `pipeline.run`. Tests with mocked models, plus one slow test running two `yoloe` jobs with different class lists in one process. `autolabel/owlv2.py` stays read-only: use `query_embedding` and `detect_image_guided` as they are. |
| backend | `mode` in #9 (`unknown_mode`, default, `config.mode`); `modes` in #16; worker: select, clean and cap the examples, order of #9, `result`, delete limited to classes that ran, store `Prediction.polygon` in detect jobs. Text jobs keep calling `run` exactly as today. Tests in `tests/test_api_*.py` / `tests/test_worker_*.py`, using `fake`. |
| ui | mode choice beside the detector (from #16; hidden when a detector has fewer than 2 modes), `mode` sent in #9, a "preparing examples" state while `running` at 0, skipped classes and example counts shown from `result`, the same in the `?mock=1` layer. |
| export | nothing: rows with `polygon` already export. |
| integrator | this contract, `docs/fixtures/`, `requirements.txt` comment, README weights step. |

### Polygons: who builds what

| Role | Builds |
|---|---|
| backend | `annotations.polygon` in `schema.sql` + `migrate()`; `polygon` in `annotation_dict`; #12 and #13 rules (validation, bbox from polygon, scaling on bbox-only PATCH); worker: store `box["polygon"]` in import jobs, pass `polygon` to `push` (`app/worker/worker.py:286`). Tests in `tests/test_api_*.py` / `tests/test_worker_*.py`; extend `tests/test_api_fixture_keys.py` to the polygon samples. |
| export | `yolo-seg` in `autolabel/export/formats.py`; `segmentation` + polygon `area` in `to_coco` and `coco_for_image`; polygon rows in `autolabel/importers/yolo.py` and `coco.py`; the per-file COCO note; the #21 test upload to `test-g9qbh`. Tests from `docs/fixtures/polygons/`. |
| ui | Spec in `docs/agents/ui.md`: show polygon outlines, draw new polygons, move / add / delete vertices, copy/paste and Repeat previous keep the polygon, undo keeps the polygon, `yolo-seg` in the export format choice, polygon rules in the `?mock=1` layer. |
| models | nothing |

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
    source: str = ""               # "owlv2", "owlv2_image", "yoloe_examples", "gdino", "pipeline", ...
    mask: dict | None = field(default=None, hash=False, compare=False)   # COCO RLE or None
    polygon: list | None = field(default=None, hash=False, compare=False)  # [[x, y], ...] 0..1, or None

@dataclass(frozen=True)
class Example:
    image_id: str
    path: str                      # image on disk
    category: str                  # class name
    bbox: tuple                    # (x, y, w, h) px, clipped to the image, area > 0

class ExampleRun(Protocol):        # one job's prototypes; built at job start, kept in memory
    classes: list[ClassSpec]       # the classes that will run, in input order
    used: dict[str, int]           # class name -> examples used (classes that run only)
    unusable: list[str]            # class names the detector rejected all examples for
    def detect(self, image: Image.Image, image_id: str) -> list[Prediction]: ...
```
`mask` and `polygon` are left out of hashing and `==` because a dict or list is not hashable (a
frozen dataclass hashes every field). Two predictions that differ only there compare equal.
```python
class Detector(Protocol):
    name: str                      # the string used in POST .../jobs "model"
    def detect(self, image: Image.Image, image_id: str,
               classes: list[ClassSpec]) -> list[Prediction]: ...
    # only detectors with mode "examples" have this one:
    def prepare_examples(self, classes: list[ClassSpec],
                         examples: dict[str, list[Example]]) -> ExampleRun: ...

# autolabel/models/__init__.py
def list_detectors() -> list[str]: ...                 # e.g. ["owlv2", "gdino", "pipeline", "fake"]
def get_detector(name: str, **options) -> Detector: ...  # loads weights lazily, once
def detector_modes(name: str) -> list[str]: ...        # default first; [] for unknown names or no modes

# autolabel/pipeline.py
def run(images: Iterable[tuple[str, str]],            # (image_id, path on disk)
        classes: list[ClassSpec],
        detector: str = "pipeline",
        options: dict | None = None,
        on_progress: Callable[[int, int], None] | None = None,   # (done, total)
        example_run: ExampleRun | None = None,                    # set: use it instead of get_detector
        ) -> Iterator[tuple[str, list[Prediction]]]: ...          # yields one image at a time

def prepare_examples(detector: str,
                     classes: list[ClassSpec],                    # classes with at least one example
                     examples: dict[str, list[Example]],          # class name -> its examples
                     options: dict | None = None) -> ExampleRun: ...

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
- When `polygon` is set, `bbox` is its bounding box in pixels.
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
---

## 5. Mock data

`docs/fixtures/api_samples.json` holds one example response per endpoint, keyed by `"METHOD path"`.
A value of `null` means 204 with an empty body (#18, #19).
It uses project 1 `eot_crane` with the 4 crane classes, 3 images, and annotations in all 5 statuses
(including one human box, one with a mask and one with a polygon).
`docs/fixtures/polygons/` holds small import fixtures: `yolo_mixed/`, `coco_segmentation/` and
`expected.json` (what `plan_import` must return for each; compare numbers with a 1e-6 tolerance).
- **ui:** add a `?mock=1` switch that reads this file instead of calling `/api`. For images, use any
  local gold-set image; the image sizes in the fixtures are 1280x720 (the real crane size).
- **export:** load `images`, `classes` and `annotations` from the fixtures and feed them to `to_coco` and `to_yolo`.
- **backend:** your real responses must match these shapes. A test that compares the keys is a good idea.
