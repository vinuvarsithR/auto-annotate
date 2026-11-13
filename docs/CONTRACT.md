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
  source      TEXT NOT NULL,              -- 'owlv2' | 'owlv2_image' | 'yoloe_examples' | 'yolo_trained' | 'gdino' | 'pipeline' | 'human' | 'import'
  status      TEXT NOT NULL CHECK (status IN
                ('auto_accepted', 'needs_review', 'accepted', 'rejected', 'edited')),
  job_id      INTEGER REFERENCES jobs(id),  -- NULL for human-drawn boxes
  reviewer    TEXT,                       -- free text name for now (no login until Phase 6)
  created_at  TEXT NOT NULL,
  updated_at  TEXT NOT NULL
);
CREATE INDEX idx_ann_image ON annotations(image_id);
CREATE INDEX idx_ann_status ON annotations(status);

CREATE TABLE trained_models (            -- #24 - #26; migrate() creates it if missing
  id           INTEGER PRIMARY KEY,
  project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  name         TEXT NOT NULL UNIQUE,     -- "eot_crane_v1"; the #9 `model` value
  version      INTEGER NOT NULL,         -- 1, 2, ... per project, never reused
  task         TEXT NOT NULL CHECK (task IN ('detect', 'segment')),
  class_names  TEXT NOT NULL,            -- JSON list, model index order
  imgsz        INTEGER,                  -- from the checkpoint, or NULL
  file_name    TEXT NOT NULL,            -- uploaded name, e.g. "best.pt"
  path         TEXT NOT NULL,            -- data/weights/<project_id>/<name>.pt
  size_bytes   INTEGER NOT NULL,
  uploaded_at  TEXT NOT NULL,
  deleted_at   TEXT,                     -- set by #26; the row is kept
  UNIQUE (project_id, version)
);

CREATE TABLE hook_points (               -- #27 - #30 (OWLv2 + SAM 2); migrate() creates it if missing
  project_id   INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  sequence     TEXT NOT NULL,            -- autolabel.sequences.sequence_of(images.file_name)
  x            REAL NOT NULL,            -- hook point in image pixels
  y            REAL NOT NULL,
  image_id     INTEGER REFERENCES images(id) ON DELETE SET NULL,  -- the image it was clicked on
  updated_at   TEXT NOT NULL,
  PRIMARY KEY (project_id, sequence)
);
-- projects gets two columns (migrate() adds them if missing; not part of the Project JSON):
--   load_classes TEXT   -- JSON list of class names, NULL = default ["rb", "sb", "biax"]
--   hook_radius  REAL   -- px, NULL = default 130
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
`yoloe` boxes (#9, examples mode), trained `segment` models and `owlv2` examples boxes from a job with
`masks: "sam2"` (OWLv2 + SAM 2 section). `polygon` and `mask` are independent; #14 and #21 use `polygon`
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
Only `class_mismatch` (#24) adds a third key, `details`.

Error codes (the UI may rely on these; `message` is for humans and may change):

| HTTP | `code` | When |
|---|---|---|
| 400 | `bad_request` | invalid body or query (incl. FastAPI validation), empty/duplicate names, unknown `status` |
| 400 | `unknown_model` | `model` not in `list_detectors()` |
| 400 | `unknown_mode` | #9 `mode` not in the detector's #16 list |
| 400 | `no_classes` | detect job on a project with no classes |
| 400 | `bad_image_ids` | `image_ids` not all in this project |
| 400 | `bad_class_id` | `class_id` not in this image's project |
| 400 | `bad_weights` | #24: the `.pt` does not load as an ultralytics checkpoint, or its task is not `detect`/`segment` |
| 400 | `task_mismatch` | #24: the checkpoint's task is not the `task` sent |
| 400 | `class_mismatch` | #24: model class names differ from the project's; the error also has `details` (see #24) |
| 404 | `not_found` | unknown id, unknown path, or image file missing on disk |
| 405 | `method_not_allowed` | wrong HTTP method for the path |
| 409 | `name_taken` | project name already used (#1 create, #17 rename) |
| 409 | `class_in_use` | PUT classes would delete a class that still has annotations |
| 409 | `conflict` | any other 409, e.g. deleting a project with a queued/running job (#18), `empty` on an image with boxes (#20), a new box on an `empty` image (#12), no `done`/`empty` images (#14, #21), Roboflow settings missing from `.env` (#21), #9 `rerun` with no examples-mode job or no `todo` image, #23 `segment` with no polygon, #26 while a job uses the model |
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
| 23 | POST | `/projects/{pid}/train-package` | download Colab training package zip |
| 24 | POST | `/projects/{pid}/weights` | upload trained `.pt` weights (multipart) |
| 25 | GET | `/projects/{pid}/weights` | list the project's uploaded models |
| 26 | DELETE | `/weights/{wid}` | delete an uploaded model |
| 27 | GET | `/projects/{pid}/hook` | hook settings + hook points per video |
| 28 | PUT | `/projects/{pid}/hook/points` | set the hook point of one video (by a clicked image) |
| 29 | DELETE | `/projects/{pid}/hook/points/{sequence}` | clear the hook point of one video |
| 30 | PATCH | `/projects/{pid}/hook` | change load classes / radius |

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
| `owlv2` | `examples` | new | `owlv2_image` | only with `masks: "sam2"` (OWLv2 + SAM 2 section) |
| `fake` | `text`, `examples` | `examples` returns its usual boxes for classes with examples | `fake` | no |

`yoloe` stays the default detector for examples mode: the one to use unless there is a reason to
try `owlv2`. This is guidance only: the `model` default in #9 and the list in #16 do not change.

Which boxes are examples:

A box is an example when (`source` is `human` AND `status` is not `rejected`) OR `status` is
`accepted` or `edited`.

| Box | Example? |
|---|---|
| `source` `human`, any status except `rejected` | yes |
| any other `source` (`import` or a model), status `accepted` or `edited` | yes |
| any other `source`, status `auto_accepted` or `needs_review` | no |
| `rejected`, any `source` | no |

So a hand-drawn box always teaches unless it was rejected, and a reviewed model or imported box
teaches the next run; an unreviewed model or imported box does not.

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
`no examples` (nothing left after the worker's clean-up), `needs at least 5 examples` (`owlv2`
only, see its rules below) or `no usable examples` (the detector rejected all of them).
`skipped_classes` is `[]` when nothing was skipped. The worker takes the reason for a class in
`ExampleRun.unusable` from `ExampleRun.skip_reasons` (section 3), and uses `no usable examples`
when the class is not there.

Scores: `score` is always 0..1, so `min_score` and `accept_score` work the same way in every mode.
They are one setting per class, shared by all modes; a value tuned for one mode is not tuned for another.

| Detector | Prototype | Score |
|---|---|---|
| `owlv2` | none: one embedding per example, all kept (no mean) | raw cosine similarity, see the `owlv2` rules below. Never `post_process_image_guided_detection`: it rescales per image, so the best box becomes 1.0 even with no object. |
| `yoloe` | one embedding per example **image** per class (all that class's example boxes in that image form one visual prompt), then the mean over images, L2-normalized, given to `set_classes(names, vpe)` | `boxes.conf` as returned; the head already applies the sigmoid. Pass `conf` = the lowest class floor, because the ultralytics default would hide low scores. |

Default floor when `min_score` is NULL: 0.05, as `owlv2` and `yolo`.

`owlv2` examples-mode rules (text mode is unchanged). Method from the models spike
(`D:\autoannotate\spikes\owl_embed.py`); threshold "rule C" from the spike report.

| Rule | Value |
|---|---|
| Box features | per image, one vision pass gives every predicted box its class embedding (L2-normalized), its box and its objectness (sigmoid). Boxes are normalized to max(W, H) (OWLv2 pads to a square). |
| Example embedding | among the image's predicted boxes with IoU >= **0.9 x** the highest IoU to the drawn box, the class embedding of the one with the highest objectness |
| Box score for a class | raw cosine similarity between the box's embedding and each of the class's example embeddings; the class score is the **max** over those examples (no averaging) |
| Fewer than 5 examples | a class with fewer than **5** examples (after the worker's clean-up and cap) is not run: it goes in `unusable`, with `skip_reasons[class] = "needs at least 5 examples"`. Fixed value, not an option. |
| Example images | assumed **fully labelled** for that class: every instance of the class in an example image has a non-rejected box (any source or status). Missing boxes raise `bg` and so the threshold. Note: **Example images must have every object of that class boxed.** (Still needed: an object with no box at all counts as background.) |
| `pos` | per class, leave-one-out: for each example, the highest cosine similarity between its embedding and the embeddings of the class's **other** examples; `pos` = the median of those values (an even count takes the mean of the two middle values). Same embeddings as the box score. |
| `bg` | per example image of the class: the highest class score on a predicted box that overlaps no labelled box of that class in that image; `bg` = the median of those values over the images. Labelled boxes = every **non-rejected** box of that class on the image, any `source` (`human`, `import`, a model) and any status (`accepted`, `edited`, `auto_accepted`, `needs_review`), including boxes left out by the 30-per-class cap: the union of `Example.bbox` (the example's own box) and `Example.labelled` (the other boxes) over the class's examples on that image. Rejected boxes count as background. "Overlaps" = IoU >= **0.3** (boxes normalized as above). Only predicted boxes with objectness >= **0.01** count. An image with no such box gives no value; when no image gives a value, the class threshold is `pos`. |
| Class threshold (rule C) | `min(pos, (pos + bg) / 2)`: halfway between `pos` and `bg`, never above `pos`. Raised to 0 if negative, so scores stay 0..1. Computed once per class in `prepare_examples`. |
| Keep a box | objectness >= **0.01** AND class score >= the class threshold AND class score >= `min_score` (or the 0.05 floor) |
| NMS | **class-agnostic**, IoU **0.3**, over all kept (box, class) candidates of the image; a box that passes for two classes keeps the higher score |
| `score` | the cosine class score |
| `min_score`, `accept_score` | apply to this cosine score, including the default floor above |
| Not used | no prototype mean, no `class_cutoff`, no calibrated sigmoid, no per-class top-5 cap |

`yoloe` rules:

| Rule | Value |
|---|---|
| Weights | `AUTOLABEL_YOLOE_WEIGHTS`, else `models/yoloe-26s-seg.pt` under the repo root (gitignored). File missing -> the job fails with a one-line error; it never downloads at job time. Must be a `-seg` file, not `-seg-pf`. |
| Enabling | `AUTOLABEL_DETECTORS` contains `yoloe`, as the other real detectors |
| Image size | 640, for example embedding and for detection |
| Model state | `set_classes` changes the model, so each job loads a fresh model in `prepare_examples` and the instance from `get_detector` keeps no job state |
| Example too small | prompts are rasterized on a 1/8 grid, and an empty prompt silently becomes a whole-image embedding. A box whose shorter side is under 8 px after scaling to 640 is not used. |
| Mask to polygon | predict with `retina_masks=True`; `cv2.findContours` with `RETR_EXTERNAL` gives the parts, and holes are dropped. A part's area is `cv2.contourArea`. **One part:** use it. **Several parts:** use the largest only if it holds **>= 90%** of the summed area of all parts; otherwise `polygon` None and the model's own bbox is kept (long beams often split, and the largest piece would shrink the box). Then simplify with `cv2.approxPolyDP`, closed, tolerance **0.001 x the image's longest side** (1.28 px at 1280); divide by width / height, round to 6 decimals |
| Bbox | with a polygon: its bounding box, computed by the detector after simplifying (section 1 bbox rule). Without one: the model's own bbox. |
| Bad polygon | fewer than 3 points or zero area -> `polygon` None, the model's own bbox is kept. `mask` is always None. |

#### Re-run with new examples (`rerun`)
After reviewers fix boxes, one call runs examples mode again with the same detector and settings
as the last time, on the images nobody has reviewed yet. It is a flag on #9, not a new endpoint:
the result is an ordinary examples-mode `detect` job (same worker path, same Job JSON, same #10
polling), so only the request and the image choice are new.

Request (nothing else may be sent):
```json
{"rerun": true}
```

| Rule | Value |
|---|---|
| `rerun` | optional boolean, default `false`. `false` or absent: #9 as above, nothing changes. Not stored in `config`. |
| Other fields with `rerun: true` | any of `model`, `mode`, `image_ids`, `options` present in the body (even `null`) -> 400 `bad_request` |
| Source job | the project's `detect` job with the highest `id` whose `config.mode` is `"examples"`, whatever its `status`. A re-run job is itself an examples job, so re-runs chain. None -> 409 `conflict` |
| Copied from the source job | `config.model`, `config.mode` (`"examples"`) and `config.options`. Not `image_ids`. |
| Checks on the copied values | as a normal #9: `unknown_model` if the detector is no longer listed, `unknown_mode` if it lost `examples`, `no_classes` |
| Images | every image of the project with `review_status` `todo`, in id order, fixed when the job is created and stored as `config.image_ids`. `done` and `empty` images are never run. None -> 409 `conflict` |
| `config` | `{"model", "mode", "image_ids", "options", "rerun_of"}`; `rerun_of` = the source job id. Only re-run jobs have `rerun_of`. |
| Examples | the usual examples rules, read when the job starts: all current examples of the project (any image, any `review_status`), cap 30 per class, skipped classes and reasons (`no examples`, `needs at least 5 examples` for `owlv2`, `no usable examples`) in `result` as usual |
| Delete before adding | the usual section 1 rule on each run image: its earlier **model** boxes still `auto_accepted` or `needs_review`, of the classes that ran, are deleted (this includes the boxes of earlier examples runs, and also of text runs), then the overlap rule. Human, imported and reviewed (`accepted`, `edited`, `rejected`) boxes are never touched. |
| Image reviewed during the job | just before writing an image, the worker reads its `review_status` again. Not `todo` any more -> nothing is deleted or added on it; it still counts in `done`. |
| `result` | the usual examples `result` (step 3 of the worker order). No new keys. |

Check order: project 404, body 400, no source job 409, `unknown_model`, `unknown_mode`,
`no_classes`, no `todo` image 409.

Response 202 is a Job, e.g. re-running job 10 on the two `todo` images 11 and 12:
```json
{"id": 11, "project_id": 1, "kind": "detect", "status": "queued", "progress": 0.0,
 "done": 0, "total": 2, "error": null,
 "config": {"model": "yoloe", "mode": "examples", "image_ids": [11, 12], "options": {}, "rerun_of": 10},
 "result": null, "created_at": "2026-10-30T15:10:00Z", "started_at": null, "finished_at": null}
```
Errors (`message` text is an example):
```json
{"error": {"code": "conflict", "message": "project 1 has no examples-mode job to re-run"}}
{"error": {"code": "conflict", "message": "project 1 has no images with review_status todo"}}
{"error": {"code": "bad_request", "message": "rerun takes no other fields; got: model"}}
```

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

### Train on Colab (#23 - #26)
The app never trains and Colab cannot reach the app, so the user carries files both ways:

| Step | Who | What |
|---|---|---|
| 1 | app (#23) | download a training package zip: reviewed images + labels, `data.yaml`, `split.json`, `train.ipynb` |
| 2 | user | open `train.ipynb` in Colab (GPU runtime), run the upload cell (picks the zip), the train cell, the download cell (saves `best.pt`) |
| 3 | app (#24) | upload `best.pt`; the server checks it loads, its task and its class names, stores it as `<project>_v<N>` |
| 4 | app (#9) | the stored model is a detector name for this project's detect jobs; #25 lists, #26 deletes |

Security: a `.pt` file is a Python pickle, and loading it runs code inside it. Upload only weights you
trained yourself. (No login until Phase 6.)

### 23. POST /projects/{pid}/train-package
Request (body optional):
```json
{"task": "segment"}
```
| Field | Default | Rule |
|---|---|---|
| `task` | `"detect"` | `detect` or `segment`; anything else, or any other field -> 400 `bad_request` |

The response is an `application/zip` download named `<project>_train_<task>.zip` (project name made
safe as in #14). Images, boxes and labels follow #14 exactly: only `review_status` `done` or `empty`
images, only the approved statuses (`auto_accepted`, `accepted`, `edited`; not configurable here),
an `empty` image gets an empty label file, zip file names from `export_names`. The image files are
always in the zip.

| `task` | Label lines | Extra rule |
|---|---|---|
| `detect` | `yolo` lines (#14) | none |
| `segment` | `yolo-seg` lines (#14): a box without polygon becomes its 4 bbox corners, clipped; no area left -> left out | at least one exported box must have a `polygon`, else 409 `conflict` |

Zip layout:
```
data.yaml
split.json
train.ipynb
images/train/<name>    labels/train/<stem>.txt
images/val/<name>      labels/val/<stem>.txt
```
`data.yaml` (the notebook unzips to `/content/dataset`, so the path is absolute and fixed):
```yaml
path: /content/dataset
train: images/train
val: images/val
nc: 4
names: ["person", "rb", "sb", "biax"]
```
`names` = project class order (#7), as in #14.

Train / val split: **by sequence, never by frame**. Sequence = `autolabel.sequences.sequence_of(file_name)`,
so every frame of one video lands in the same split.

| Rule | Value |
|---|---|
| Target | `T = max(1, round(0.2 x N))` val images, N = exported images |
| Order | sequences by image count ascending, then by name |
| Pick | walk that order; a sequence goes to val when (a) val images + its images <= `T`, (b) at least one sequence stays in train, and (c) every class with >= 1 exported box keeps >= 1 box in train |
| Nothing picked | the first sequence in the order that meets (b) and (c) goes to val alone; if none meets (c), the first that meets (b) (then `classes_missing_in_train` is not empty) |
| One sequence only | no split possible: every image is in `images/train`, `data.yaml` has `val: images/train`, `val_is_train: true`; there is no `images/val` or `labels/val` |

`split.json`:
```json
{"rule": "by_sequence", "val_target": 0.2, "val_is_train": false,
 "train": {"sequences": ["biax__cam0_2026-09-29_18-48-25", "rb_load__cam0_2026-09-29_18-34-11"], "images": 40},
 "val": {"sequences": ["rb_load__cam0_2026-09-29_18-24-11"], "images": 10},
 "classes_missing_in_train": [], "classes_missing_in_val": ["sb"]}
```
Sequence lists are sorted by name. With `val_is_train: true`, `val` = `{"sequences": [], "images": 0}`.
`classes_missing_in_*` list class names (project order) that have exported boxes but none in that split
(with `val_is_train`, `classes_missing_in_val` is `[]`).

`train.ipynb`: nbformat 4 JSON (`"nbformat": 4`), kernel `python3`, `"metadata"` has `"accelerator": "GPU"`,
exactly **4 cells**, in this order:

| # | Type | Must contain |
|---|---|---|
| 1 | markdown | project name, task, the split counts, "Runtime > Change runtime type > GPU", and the 3 steps below |
| 2 | code (upload) | `!pip install -q ultralytics==8.4.199` (the version in `requirements.txt`); `from google.colab import files`; `files.upload()`; removes then fills `/content/dataset` by unzipping the uploaded zip |
| 3 | code (train) | `YOLO("yolo11s-seg.pt")` for `segment`, `YOLO("yolo11s.pt")` for `detect`; `.train(data="/content/dataset/data.yaml", imgsz=960, epochs=100, project="/content/runs", name="<run>", exist_ok=True)` |
| 4 | code (download) | copies `/content/runs/<run>/weights/best.pt` to `/content/<run>_best.pt` and calls `files.download("/content/<run>_best.pt")` |

`<run>` = `<safe project name>_<task>`. Defaults match the earlier Colab run (`yolo11s`, imgsz 960, file
`yolo11s_960_v2.pt`). `epochs=100` is the ultralytics default; the earlier run's epoch count is not
recorded (GUESS). Every code cell's `outputs` is `[]` and `execution_count` is `null`.

Errors (in this order): unknown pid -> 404 `not_found`; bad body or `task` -> 400 `bad_request`; no
`done`/`empty` image -> 409 `conflict`; `segment` with no exported polygon -> 409 `conflict`; an
exported image's file missing on disk -> 404 `not_found`. No zip is sent on an error.

Route: `app/api/export.py` (same file as #14; mounted under `/api`). It loads rows and calls
`build_train_package` (section 3, training package).

### 24. POST /projects/{pid}/weights
Upload trained weights. `multipart/form-data`:

| Part | Rule |
|---|---|
| `file` | exactly one, file name ending `.pt` (any case), 1 byte .. **200 MB** |
| `task` | form field, required: `detect` or `segment` = what the user trained (the package's task) |

Checks, in this order (nothing is stored when one fails; the temporary file is deleted):

| Check | Error |
|---|---|
| unknown pid | 404 `not_found` |
| missing / extra parts, not `.pt`, empty, > 200 MB, bad `task` | 400 `bad_request` |
| project has no classes | 400 `no_classes` |
| `inspect_weights(path)` raises (not a loadable ultralytics checkpoint) | 400 `bad_weights` |
| `info.task` is not `detect` or `segment` (e.g. `pose`, `classify`) | 400 `bad_weights` |
| `info.task` != `task` | 400 `task_mismatch` |
| model class names != project class names (as sets, exact case; or a repeated model name) | 400 `class_mismatch`, with `details` |

Class names are compared **by name, never remapped**. Order may differ (detection matches by name,
as `yolo`). `class_mismatch` is the one error with an extra key:
```json
{"error": {"code": "class_mismatch",
           "message": "model classes do not match project classes: missing in model: sb; extra in model: beam",
           "details": {"model_classes": ["person", "rb", "beam", "biax"],
                       "project_classes": ["person", "rb", "sb", "biax"],
                       "missing_in_model": ["sb"], "extra_in_model": ["beam"]}}}
```
(`missing_in_model` in project order, `extra_in_model` in model order; a repeated model name is listed
once in `extra_in_model`.)

Storage: `data/weights/<pid>/<name>.pt` (under the app data dir; `data/` is gitignored).

| Field | Rule |
|---|---|
| `version` | 1 + the highest `version` this project ever had, deleted rows included, so a number is never reused |
| `name` | `<safe project name>_v<version>`, e.g. `eot_crane_v1` (project name made safe as in #14, at upload time; later renames do not change it). If another project already uses that name: `<safe project name>_p<pid>_v<version>` |

Response 201 is a **TrainedModel**:
```json
{"id": 1, "project_id": 1, "name": "eot_crane_v1", "version": 1, "task": "segment",
 "class_names": ["person", "rb", "sb", "biax"], "imgsz": 960, "file_name": "best.pt",
 "size_bytes": 19876543, "uploaded_at": "2026-11-11T10:00:00Z"}
```
`class_names` = the model's names in model index order; `imgsz` = from the checkpoint or `null`.
Owner: backend (route, storage, checks); `inspect_weights` comes from models.

### 25. GET /projects/{pid}/weights
Response `{"weights": [TrainedModel, ...]}`: the project's uploaded models that are not deleted,
by `version` ascending. Unknown pid -> 404 `not_found`. Owner: backend.

### 26. DELETE /weights/{wid}
No body. Response **204**. Deletes the `.pt` file, sets `deleted_at` (the row stays so the version is
never reused and old jobs still name a real model), and calls `unregister_trained(name)`. After that
the name is gone from #16 and #25, and #9 with it -> 400 `unknown_model`. A queued or running detect job
whose `config.model` is this name -> 409 `conflict`, nothing changes. Unknown or already deleted
`wid` -> 404 `not_found`. Owner: backend.

### Train on Colab: changes to #9, #16, #18

| Entry | Change |
|---|---|
| #9 | `model` may also be the `name` of a non-deleted TrainedModel **of this project** (another project's name -> 400 `unknown_model`). It has no modes, so `mode` must be absent or `null` (else `unknown_mode`). The job runs like a text job. Worker: before `run`, look up the row (deleted or missing -> the job fails, `error` "model NAME was deleted"), call `register_trained(name, path, task)`, then call `run(..., detector=name, options=config.options)` exactly as for any detector. |
| #16 | optional query `project_id`. Given: `models` = `list_detectors()` + that project's non-deleted TrainedModel names (version order), and each of those has `modes[name] = []`. Unknown `project_id` -> 404 `not_found`. Absent: unchanged. |
| #18 | also deletes `data/weights/<pid>/` and calls `unregister_trained` for each of the project's names. The project's `trained_models` rows (deleted ones too) are deleted with it, like its other child rows. |

Who builds what (Train on Colab):

| Role | Builds |
|---|---|
| export | `autolabel/export/train_package.py`: `build_train_package` (zip, `data.yaml`, `split.json`, split rule, labels via the #14 functions) and the notebook template; `NoPolygons`; route #23 in `app/api/export.py`. Must pass `tests/test_export_train_package.py`. |
| models | `autolabel/models/trained.py`: `inspect_weights`, `register_trained`, `unregister_trained`, the trained detector (section 3); `get_detector` / `detector_modes` for registered names. Must pass `tests/test_models_trained.py`. |
| backend | `trained_models` table + `migrate()`; #24, #25, #26; the #9, #16, #18 changes; worker step for trained names; `create_app(..., inspect_weights=None, register_trained=None, unregister_trained=None)` (None = the `autolabel.models` functions, imported lazily). Must pass `tests/test_api_trained_models.py`. |
| ui | a **Train** panel: task choice (`segment` only offered when the project has a polygon), "Download training package" (#23), short Colab steps, `.pt` upload with task (#24) showing `class_mismatch` details as two lists, the model list with delete (#25, #26), model choice in the detect form from #16 `?project_id=`; the same in `?mock=1`. |
| integrator | this contract, fixtures, the three test files above (written first; roles may add tests, not weaken these). |

---

### Examples mode: who builds what

| Role | Builds |
|---|---|
| models | `autolabel/models/yoloe.py` (weights, `prepare_examples`, `mask_to_polygon` as a pure function); `prepare_examples` on `owlv2` and `fake`; `Example`, `ExampleRun`, `Prediction.polygon` in `base.py`; `detector_modes` and the `yoloe` registry entry; `pipeline.prepare_examples` and the `example_run` argument of `pipeline.run`. Tests with mocked models, plus one slow test running two `yoloe` jobs with different class lists in one process. `owlv2` examples-mode rules in `autolabel/models/owlv2.py` (example embedding, cosine score, 5-example minimum, rule C threshold, class-agnostic NMS), and `skip_reasons` on its `ExampleRun`. `autolabel/owlv2.py`: models may add one method returning per-box class embeddings, boxes and objectness (it may replace the logit method models added); existing functions and the text path stay unchanged. |
| backend (owlv2 examples change) | worker: reason from `ExampleRun.skip_reasons` (read with `getattr(run, "skip_reasons", {})`, so `fake` and `yoloe` need no change), else `no usable examples`. Test with a fake run giving `needs at least 5 examples`. Worker fills `Example.labelled` (section 3) for every example: the **other** non-rejected boxes of that class on that image (not the example's own box), any source or status, before the cap, same clipping as clean-up. Test: a capped-out box and an `auto_accepted` box appear in `labelled`, a `rejected` one does not. |
| models (owlv2 bg change) | `bg` uses the union of `Example.bbox` and `Example.labelled` per image (works with `labelled` empty). Adds the `labelled` field to `Example` in `base.py`. Test with a non-example box in `labelled` that would otherwise set `bg`. |
| backend | `mode` in #9 (`unknown_mode`, default, `config.mode`); `modes` in #16; worker: select, clean and cap the examples, order of #9, `result`, delete limited to classes that ran, store `Prediction.polygon` in detect jobs. Text jobs keep calling `run` exactly as today. Tests in `tests/test_api_*.py` / `tests/test_worker_*.py`, using `fake`. |
| ui | mode choice beside the detector (from #16; hidden when a detector has fewer than 2 modes), `mode` sent in #9, a "preparing examples" state while `running` at 0, skipped classes and example counts shown from `result`, the same in the `?mock=1` layer. |
| export | nothing: rows with `polygon` already export. |
| integrator | this contract, `docs/fixtures/`, `requirements.txt` comment, README weights step. |
| backend (`rerun`) | #9 `rerun` flag: body check (`model_fields_set`), source job, copied config, `todo` image list, `rerun_of`, the two 409s; worker re-reads `review_status` before writing each image of any job whose config has `rerun_of`. Must pass `tests/test_api_rerun_examples.py` (written by the integrator; backend may add tests, not weaken these). |
| ui (`rerun`) | a "Re-run with new examples" button, enabled once the project has an examples-mode job; sends `{"rerun": true}`, then polls like any job; shows the two 409 messages; the same in `?mock=1`. |
| models, export (`rerun`) | nothing |

### Polygons: who builds what

| Role | Builds |
|---|---|
| backend | `annotations.polygon` in `schema.sql` + `migrate()`; `polygon` in `annotation_dict`; #12 and #13 rules (validation, bbox from polygon, scaling on bbox-only PATCH); worker: store `box["polygon"]` in import jobs, pass `polygon` to `push` (`app/worker/worker.py:286`). Tests in `tests/test_api_*.py` / `tests/test_worker_*.py`; extend `tests/test_api_fixture_keys.py` to the polygon samples. |
| export | `yolo-seg` in `autolabel/export/formats.py`; `segmentation` + polygon `area` in `to_coco` and `coco_for_image`; polygon rows in `autolabel/importers/yolo.py` and `coco.py`; the per-file COCO note; the #21 test upload to `test-g9qbh`. Tests from `docs/fixtures/polygons/`. |
| ui | Spec in `docs/agents/ui.md`: show polygon outlines, draw new polygons, move / add / delete vertices, copy/paste and Repeat previous keep the polygon, undo keeps the polygon, `yolo-seg` in the export format choice, polygon rules in the `?mock=1` layer. |
| models | nothing |

### OWLv2 + SAM 2 polygons (#27 - #30, #9 `masks`)
Approved 2026-11-13 from spike v2 (`D:\autoannotate\spikes\owl_sam_spike_v2\results_v2.md`,
`choices.json`; all values picked on train frames, scored on 40 test frames). After `owlv2` examples-mode
detection, four steps run in this order, per image:

| # | Step | Runs when | Spike effect on test (all classes, VERIFIED) |
|---|---|---|---|
| 1 | containment merge | every `owlv2` examples job | box F1 0.52 -> 0.55 |
| 2 | hook filter + one load per image | every `owlv2` examples job, on images whose video has a hook point | 0.55 -> 0.73 (63 load boxes dropped, 0 true positives lost) |
| 3 | SAM 2.1 tiny mask from the box | `masks: "sam2"` | mask F1 0.72 -> 0.73 |
| 4 | mask -> one polygon, no box fallback | `masks: "sam2"` | see step 4 (largest part: test mask F1 0.72) |

Text mode, other detectors and other modes are unchanged. Speed (spike v1, Windows CPU, VERIFIED): OWLv2
4.59 s/img + SAM 2.1 tiny 1.96 s/img.

**Step 1, containment merge** (`merge_contained`). Two boxes of the **same class**, where the one with
the smaller-or-equal area has **>= 80%** of its area inside the other (intersection / smaller area >= 0.8)
-> one box: the **union** of the two bboxes, `score` = the higher of the two, every other field from the
larger box. Repeat until no pair is left. Scan order (makes results exact): for `i` in list order, for `j`
in list order, `j != i`, the first pair where `j` is the smaller-or-equal box and the rule holds: the merged
box replaces item `i` in place, item `j` is removed, start again. Different classes never merge.

**Step 2, hook filter** (`hook_filter`). Per video (= `sequence_of(images.file_name)`, the same key as the
#23 split), the project may have one hook point (x, y px, set by a click in the UI). Load classes =
project setting (#30), default `["rb", "sb", "biax"]`; names that are not project classes are ignored.

| Rule | Value |
|---|---|
| Distance | center of the bbox to the hook point, Euclidean px. A load box with distance **> radius** is dropped; `<=` is kept. Radius = project setting (#30), default **130** px. |
| One load per image | after the distance rule, keep only the load box with the highest `score` (on a tie, the first in list order); all load classes count together. Non-load boxes are never touched. |
| No hook point for the image's video | nothing is filtered on that image (neither rule), and the job `result` gets a warning for that video (below) |
| Load classes empty | the step is off: nothing filtered, no warnings |
| When read | hook points and settings are read once at job start (worker step 1) and fixed for the job |

**Step 3, SAM mask** (`Segmenter`, `clean_mask`).

| Rule | Value |
|---|---|
| Model | SAM 2.1 tiny through ultralytics (`ultralytics==8.4.199`, as in `requirements.txt`), CPU. Weights: `AUTOLABEL_SAM_WEIGHTS`, else `models/sam2.1_t.pt` under the repo root (gitignored). Missing file -> the job fails with a one-line error before the first image; it never downloads. Loaded once per process, lazily. |
| Prompt | the box (after steps 1-2), pixels, one call per image for all its boxes, `multimask_output=True` (3 masks + 3 scores per box), masks at image size. The spike's call path is `predictor.set_image` + `predictor.inference_features(..., bboxes=..., multimask_output=True)` (`steps.py`, VERIFIED on 8.4.199); models may use any call that gives the same masks. |
| Pick | among the box's masks that are non-empty and have **>= 90%** of their pixels inside the box grown by **5 px** on each side: the one with the highest SAM score. None qualifies -> the mask with the highest inside fraction. |
| Opening | `cv2.morphologyEx(MORPH_OPEN)` with a **5 x 5** `MORPH_ELLIPSE` kernel. If that empties the mask, keep the mask from before the opening. |
| Clip | pixels outside the box (`floor(x0)..ceil(x1)`, `floor(y0)..ceil(y1)`, inclusive) are cleared |

**Step 4, mask -> polygon: single largest part** (`largest_polygon`). `cv2.findContours(RETR_EXTERNAL)`
gives the parts (holes dropped); the part with the largest `cv2.contourArea` is used, **always** (no 90%
rule, unlike `yoloe`). Simplify as `yoloe` (`approxPolyDP`, closed, tolerance 0.001 x the image's longest
side), divide by width / height, round to 6 decimals. Then `polygon` = that ring and `bbox` = its bounding
box (section 1 rule), `score`, `category` and `source` (`owlv2_image`) unchanged, `mask` None.
**No box fallback:** an empty mask, or a polygon with fewer than 3 points or zero area -> the box is
dropped (not stored).

Why one part and not multi-part (spike chose multi-part):

| Where the polygon goes | One ring | Several rings |
|---|---|---|
| `annotations.polygon`, API, section conventions | yes (the convention is "one outer ring") | needs a new shape and migration |
| #14 / #23 `yolo-seg` (one line = one object, one ring) | yes | no: two lines would be two objects |
| #14 COCO `segmentation` | yes | yes (COCO allows a list of parts) |
| #21 Roboflow push (`coco_for_image`, one ring) | yes, tested 2026-11-04 | not tested (GUESS) |
| #22 import of our own COCO export | yes | no: multi-part is imported as box only |
| UI polygon editor | yes | no |

Cost on the spike (VERIFIED, `results_v2.md`): train frames tie (mask F1 0.706 both); test frames mask
F1 0.72 vs 0.75 and box F1 0.71 vs 0.74 (the box shrinks to the largest part when a beam splits). Small
sample: one sb or biax object moves F1 by about 0.1. Multi-part can come later as its own change.

#### #9 change: `masks`
Request, examples mode with `owlv2`:
```json
{"model": "owlv2", "mode": "examples", "image_ids": null, "options": {}, "masks": "sam2"}
```

| Rule | Value |
|---|---|
| `masks` | optional top-level field (like `mode`, never in `options`): `"sam2"` or `"none"`. Absent = `"none"`. Any other value -> 400 `bad_request`. |
| Allowed for | resolved `model` `owlv2` and `mode` `examples` only; `masks` present (any value) on another model or mode -> 400 `bad_request` |
| `config.masks` | stored only when the request sent it, so older jobs and other jobs keep their `config` shape |
| `rerun` | also copies `config.masks` from the source job when present (`rerun` itself still takes no other field) |
| Check order | after `unknown_mode`, before `no_classes` |

`result` of an `owlv2` examples job (written at worker step 3, as the other keys) gets two more keys:
```json
{"mode": "examples", "examples_by_class": {"person": 30, "rb": 12, "biax": 6},
 "skipped_classes": [{"class_name": "sb", "reason": "needs at least 5 examples"}],
 "masks": "sam2",
 "hook": {"load_classes": ["rb", "sb", "biax"], "radius": 130.0,
          "points": {"rb_load__cam0_2026-09-29_18-34-11": [616.0, 347.0]},
          "warnings": [{"sequence": "biax__cam0_2026-09-29_18-48-25", "images": 1, "reason": "no hook point"}]}}
```

| Key | Rule |
|---|---|
| `masks` | `"sam2"` or `"none"` (the resolved value) |
| `hook.load_classes`, `hook.radius` | the effective settings used (defaults filled in) |
| `hook.points` | sequence -> `[x, y]`, only for sequences that have at least one image in this job |
| `hook.warnings` | one entry per sequence with images in this job and no hook point, sorted by `sequence`; `images` = how many of this job's images are in it; `reason` is always `"no hook point"`. `[]` when no load class ran in this job, or load classes are empty. |

Other detectors' `result` keep exactly the keys of #9 (no `masks`, no `hook`).

#### 27. GET /projects/{pid}/hook
Response 200 **HookSettings**:
```json
{"load_classes": ["rb", "sb", "biax"], "load_classes_default": true, "radius": 130.0, "radius_default": true,
 "points": [{"sequence": "rb_load__cam0_2026-09-29_18-34-11", "x": 616.0, "y": 347.0, "image_id": 10,
             "updated_at": "2026-11-13T10:00:00Z"}],
 "sequences": [{"sequence": "biax__cam0_2026-09-29_18-48-25", "images": 1, "has_point": false},
               {"sequence": "rb_load__cam0_2026-09-29_18-34-11", "images": 1, "has_point": true},
               {"sequence": "sb_evener__cam0_2026-09-29_18-57-19", "images": 1, "has_point": false}]}
```
`points` sorted by `sequence`; `image_id` may be `null` (image gone). `sequences` = every sequence of the
project's images, sorted, with its image count. A point may exist for a sequence with no images left; it
is listed in `points` only. `*_default` = the value is the default (the DB column is NULL). Unknown pid -> 404.

#### 28. PUT /projects/{pid}/hook/points
```json
{"image_id": 10, "x": 616, "y": 347}
```
Sets (or replaces) the hook point of that image's video: `sequence = sequence_of(image.file_name)`,
`image_id` stored. Response 200 is the point (one item of #27 `points`). Errors, in order: unknown pid ->
404 `not_found`; bad body (missing field, extra field, `x`/`y` not finite numbers) -> 400 `bad_request`;
image not in this project -> 400 `bad_image_ids`; `x` outside `0..width` or `y` outside `0..height` of that
image -> 400 `bad_request`.

#### 29. DELETE /projects/{pid}/hook/points/{sequence}
`sequence` is URL-encoded. Response **204**. Unknown pid, or no point for that sequence -> 404 `not_found`.

#### 30. PATCH /projects/{pid}/hook
```json
{"load_classes": ["rb", "sb", "biax"], "radius": 120}
```
Both optional; `null` resets that one to its default. `load_classes`: list of distinct current class
names of the project (`[]` allowed = filter off); an unknown or repeated name -> 400 `bad_request`.
`radius`: finite number > 0, else 400 `bad_request`. Extra fields -> 400 `bad_request`. Response 200 is
HookSettings (#27). A later class delete (#8) does not change the setting; deleted names are ignored at
job time.

#18 also removes the project's `hook_points` rows (ON DELETE CASCADE).

#### Python interface (owner: models)
```python
# autolabel/models/base.py
@dataclass(frozen=True)
class PostProcess:                 # worker -> pipeline.run, one per job
    load_classes: tuple[str, ...] = ("rb", "sb", "biax")
    radius: float = 130.0
    hooks: dict = field(default_factory=dict, hash=False, compare=False)
                                   # image_id (str, as passed to run) -> (x, y) px; missing = no hook filter
    masks: str = "none"            # "sam2" | "none"

# autolabel/models/postprocess.py (steps 1-2, pure, no torch)
CONTAIN_FRAC = 0.8
def merge_contained(preds: list[Prediction], frac: float = CONTAIN_FRAC) -> list[Prediction]: ...
def hook_filter(preds: list[Prediction], hook: tuple[float, float] | None,
                load_classes: Iterable[str], radius: float) -> list[Prediction]: ...

# autolabel/models/sam.py (steps 3-4)
class Segmenter(Protocol):
    def segment(self, image: Image.Image, boxes: list[tuple]) -> list[tuple[np.ndarray, np.ndarray]]: ...
    # boxes (x, y, w, h) px; per box: (masks bool k x H x W at image size, scores float k), k >= 1
def get_segmenter() -> Segmenter: ...      # SAM 2.1 tiny, cached per process; missing weights ->
                                           # FileNotFoundError with a one-line message
class FakeSegmenter:                       # tests: k = 1, the filled ellipse inscribed in the box, score 0.9
    ...
def clean_mask(masks, scores, bbox: tuple, open_px: int = 5) -> np.ndarray: ...   # step 3 pick/open/clip
def largest_polygon(mask: np.ndarray, width: int, height: int) -> list | None: ...  # step 4; None = drop
def add_polygons(image: Image.Image, preds: list[Prediction], segmenter: Segmenter) -> list[Prediction]: ...

# autolabel/pipeline.py: run() gets two keyword arguments
def run(..., post: PostProcess | None = None, segmenter: Segmenter | None = None): ...
    # post set: per image, after detection: merge_contained -> hook_filter(hooks.get(image_id)) ->
    # if post.masks == "sam2": add_polygons (segmenter, else get_segmenter(), fetched before the
    # first image). post None: exactly as today. Applies to whatever detector runs (tests use fake).
```
The worker passes `post` only for `owlv2` examples jobs; then the usual delete + `drop_overlapping`
(section 1) runs on what `run` yields.

#### Who builds what (OWLv2 + SAM 2)

| Role | Builds | Tests (must pass) |
|---|---|---|
| models | `PostProcess` in `base.py`; `autolabel/models/postprocess.py`; `autolabel/models/sam.py` (`get_segmenter`, `FakeSegmenter`, `clean_mask`, `largest_polygon`, `add_polygons`); `pipeline.run(post=, segmenter=)` | `tests/test_models_postprocess.py`: every case in `docs/fixtures/owl_sam/cases.json` `merge` and `hook_filter`. `tests/test_models_sam.py` with `FakeSegmenter` and numpy masks only (no weights): multimask pick (inside rule + fallback), opening, opening that empties, clip, two-part mask -> largest part + bbox, empty mask -> box dropped, missing weights -> `FileNotFoundError`. `tests/test_pipeline_*.py`: `run(post=...)` with `fake` + `FakeSegmenter`, order of the steps. |
| backend | `hook_points` table, the two `projects` columns, `migrate()`; #27 - #30; #9 `masks` (checks, `config.masks`, `rerun` copy); worker: read settings + points at job start, build `PostProcess` (hooks by image id via `sequence_of`), pass it for `owlv2` examples jobs, `result.masks` + `result.hook` (warnings). Tests use the existing fake pipeline, no SAM. | `tests/test_api_hook.py` (#27 - #30 incl. errors, keys = fixtures), `tests/test_worker_*.py`: `cases.json` `hook_warnings`, `PostProcess` passed only for `owlv2` examples, `masks` 400s, `rerun` copies `masks` |
| ui | examples mode with `owlv2`: a "SAM 2 masks" toggle (sends `masks`); on the review page, a "Set hook point" action: click on the image -> #28 with image-pixel coords, draws the point and the radius circle, "Clear" -> #29; shows `result.hook.warnings` after a job; `?mock=1` from the fixtures | manual check |
| export | nothing: rows have one polygon ring, as today | - |
| integrator | this section, `docs/fixtures/owl_sam/cases.json`, fixtures #27 - #30 + job 12 in `api_samples.json` | full `pytest` at merge |

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
    labelled: tuple = ()           # the OTHER non-rejected boxes of this class on this image (not this
                                   # example's own box), any source or status, cap ignored,
                                   # as (x, y, w, h) px clipped, area > 0 (owlv2 bg)

class ExampleRun(Protocol):        # one job's prototypes; built at job start, kept in memory
    classes: list[ClassSpec]       # the classes that will run, in input order
    used: dict[str, int]           # class name -> examples used (classes that run only)
    unusable: list[str]            # class names the detector rejected all examples for
    skip_reasons: dict[str, str]   # optional: class in `unusable` -> reason for #9 `result`
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

Trained weights (#24 - #26; owner: models, `autolabel/models/trained.py`, re-exported from
`autolabel.models`):
```python
@dataclass(frozen=True)
class WeightsInfo:
    task: str                      # the checkpoint's ultralytics task: "detect", "segment", "pose", ...
    class_names: list[str]         # model names in index order
    imgsz: int | None              # train_args imgsz (as yolo.trained_imgsz), else None

def inspect_weights(path: str, model=None) -> WeightsInfo: ...
    # loads with ultralytics (autoinstall off, CPU, never downloads); `model` = a stand-in with
    # .task, .names (dict or list) and .ckpt / .overrides, for tests (then `path` is not opened).
    # Any load failure -> ValueError with a one-line message. Does not judge the task.
def register_trained(name: str, path: str, task: str, model=None) -> None: ...
    # after this, get_detector(name) returns the trained detector (built lazily, once per name),
    # whatever AUTOLABEL_DETECTORS says; list_detectors() does NOT include it; detector_modes(name) = [].
    # Same name + same path again: no-op (cache kept). A different path replaces it (cache dropped).
    # `model` = a stand-in used instead of loading `path`, for tests.
def unregister_trained(name: str) -> None: ...   # forgets it and its cached instance; unknown name: no-op
```
Trained detector: the `yolo` rules (classes by name, `min_score` floors with 0.05 default, `imgsz` from
the checkpoint, CPU, autoinstall off), `source` = `"yolo_trained"`. Task `segment`: predict with
`retina_masks=True` and set `polygon` from each box's mask (`result.masks.data[i]`, H x W at image size,
tensor or numpy) with `yoloe.mask_to_polygon` (same rules as
`yoloe`: one part, or the largest if >= 90%, else `polygon` None and the model bbox); with a polygon the
bbox is its bounding box. Task `detect`: `polygon` None. `mask` is always None.

Training package (#23; owner: export, `autolabel/export/train_package.py`, re-exported from
`autolabel.export`):
```python
TRAIN_TASKS = ("detect", "segment")

class NoPolygons(Exception): ...    # segment asked, no exported box has a polygon (route -> 409)

def build_train_package(project_name: str, task: str,
                        images: list[dict], classes: list[dict], annotations: list[dict],
                        image_bytes: Callable[[dict], bytes]) -> bytes: ...
    # API JSON shapes in (as build_zip); returns the zip of #23. Raises ValueError (bad task),
    # NothingToExport (no done/empty image), NoPolygons. Calls image_bytes only for exported images.
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
  everyone. Import them, wrap them, or ask the integrator. One exception: models may add
  one method to `autolabel/owlv2.py` for examples mode (see *Examples mode: who builds what*).
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
