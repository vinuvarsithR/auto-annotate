# backend session notes

Build against docs/CONTRACT.md. Only edit files this session owns (CONTRACT.md section 4).

## Done

Sections 1 (SQLite) and 2 (REST, endpoints 1-13) of the contract. 14 and 15 skipped as asked.

| File | What |
|---|---|
| `app/main.py` | `create_app(data_dir="data", start_worker=True, load_pipeline=None, list_detectors=None, static_dir=app/static)`; module-level `app` for `uvicorn app.main:app --port 8000`. Tables created on startup (lifespan), not on import |
| `app/db/schema.sql`, `app/db/db.py` | contract schema verbatim; connect (foreign keys ON, WAL), row -> JSON shapes, counts |
| `app/api/errors.py` | `ApiError`; all errors (incl. FastAPI validation -> 400 `bad_request`, unknown path -> 404) use `{"error": {code, message}}` |
| `app/api/deps.py` | one SQLite connection per request |
| `app/api/{projects,images,classes,jobs,annotations}.py` | endpoints 1-3, 4-6, 7-8, 9-10, 11-13 |
| `app/worker/worker.py` | one background thread; `run_once()` for tests |
| `app/api/models.py` | endpoint 16 `GET /api/models` -> `{"models": app.state.list_detectors()}` (2026-10-31). Tests: `test_api_models.py` (2) + fixture key check; 172 total pass (VERIFIED, Windows) |
| `app/api/projects.py`, `app/api/annotations.py` | 2026-10-31: #17 rename, #18 delete project (409 `conflict` if a job is queued/running; explicit child deletes in one `BEGIN IMMEDIATE` transaction, then removes `data/uploads/<pid>/`; if that removal fails after the rows are gone: warning in the log, still 204), #19 hard-delete box, #13 explicit status wins. 180 tests pass (VERIFIED, Windows) |
| tests | `test_api_{projects,images,classes,jobs,annotations,fixture_keys,helpers}.py`, `test_db_schema.py`, `test_worker_status.py`: 45 new tests, 152 total pass (VERIFIED, Windows, Python 3.14.0) |

Behaviour choices (approved in the plan):

| Case | Behaviour |
|---|---|
| PATCH with an explicit `status` (any of the 5) | that status, even if bbox/mask/class changed too (contract #13, 2026-10-31). Without `status`, a real change gives `edited`; sending the same bbox again is not an edit |
| PATCH unknown `status` | 400 `bad_request` |
| Model box below class `min_score` | dropped by the worker too (safety net) |
| Detector returns unknown class / image_id | job `failed`, that image's rows untouched |
| Jobs still `running` at startup | marked `failed` ("server restarted while the job was running") |
| Job with no classes / foreign image ids / unknown model | 400 `no_classes` / `bad_image_ids` / `unknown_model` |
| Upload | PIL must open it as JPEG/PNG **and** the extension must match, else `not an image`; name reduced to basename |
| Re-run detect | per image, when its results arrive: delete that image's non-human `auto_accepted`/`needs_review` rows, insert new ones, update progress, in one transaction. A failed job leaves images it never reached unchanged |
| `config` stored on a job | the request body after defaults: `{"model", "image_ids", "options"}` |

Worker loads the models code lazily: `from autolabel import pipeline` and
`from autolabel.models.base import ClassSpec`, then calls
`pipeline.run(images=[(str(image.id), path)], classes=[ClassSpec], detector=model, options=options)`
(on_progress not passed; the worker updates progress per yielded image).
`POST .../jobs` checks the model against `autolabel.models.list_detectors()`. Until the
models branch is merged, this returns 400 `unknown_model` ("autolabel.models is not installed").
Tests use their own fake (`tests/test_api_helpers.py`: FakeClassSpec/FakePrediction/FakePipeline).

### #21 backend part (B1-B3, 2026-11-01)

| Task | What | Where |
|---|---|---|
| B1 | `images.roboflow_image_id TEXT` in schema.sql; `migrate()` adds it to old DBs (rows kept, value NULL). Not in Image JSON | `app/db/` |
| B2 | Worker takes any queued job (oldest first). `roboflow_push`: images `done`/`empty` with NULL id, boxes with status in `config.statuses` (`empty` image -> no boxes), batch `autolabel_<started_at date>`, id saved per image, `failed` + one-line error with the key replaced by `***` | `app/worker/worker.py` |
| B3 | `db.create_job(conn, pid, kind, config, total, wake=None) -> Job`. `total` 0 -> stored `done` at once (no wake). Route passes `wake=request.app.state.worker.wake` | `app/db/db.py` |

The #9 detect route is unchanged (still inserts its own row), so detect behaviour did not change.
Tests: `tests/test_worker_push.py`, migration tests in `tests/test_db_schema.py`. Pusher faked, no network.

### #22 import + Job.result (2026-11-03)

| What | Where |
|---|---|
| Route `POST /projects/{pid}/import`: 404, then 400 for no files / bad zip / unsafe or repeated path / above `AUTOLABEL_IMPORT_MAX_GB`. Upload saved in `data/imports/staging_*`, moved to `data/imports/<job_id>/` inside the job's insert transaction (`create_job(..., before_commit=)`), so the worker never sees a job without files | `app/api/imports.py` |
| Worker `import` jobs: extract zip, `plan_import`, add missing classes at the end, per image one transaction (#4 rule via shared `db.insert_image`, boxes `needs_review`/`import`), `result`, folder deleted in `finally` | `app/worker/worker.py` |
| `jobs.result` + kind `'import'`: `migrate_jobs()` rebuilds `jobs` (FK off, one transaction, row count + new-FK-problem checks, rollback on error, ids kept). Verified on copies of both real old DBs (auto-annotate 10 jobs, aa-ui 1 job): every row kept | `app/db/db.py` |
| Detect jobs no longer delete `source='import'` boxes (section 1 rule) | worker |
| Startup deletes `data/imports/*` folders no queued job needs | `app/main.py` |

Importers are loaded lazily (`Worker.load_importers`, default `autolabel.importers`); tests use
`tests/test_api_import_stub.py`. Until export merges, the live route returns 500 "autolabel.importers is not installed".
Not done here: the #9 `drop_overlapping` call (waits for models' `pipeline.drop_overlapping`).
Note: python-multipart cuts a part name like `C:\x\a.jpg` to `a.jpg` before the route sees it.
An import job with 0 images is still queued (not stored `done` at once) so it writes `result` and cleans up.

## Packages needed
<!-- package, version, why, VERIFIED or GUESS. The integrator adds them to requirements.txt. -->

| Package | Version | Why | |
|---|---|---|---|
| fastapi | 0.162.0 | API | VERIFIED installed (import, 2026-10-31, Windows .venv) |
| uvicorn | 0.54.0 | server on port 8000 | VERIFIED installed |
| python-multipart | 0.0.32 | multipart image upload | VERIFIED installed |
| httpx | 0.28.1 | FastAPI TestClient in tests | VERIFIED installed |
| Pillow | 13.0.0 | image size / format check on upload | VERIFIED installed (already used by autolabel) |

## Contract change requests

1. **Export router hookup.** `app/main.py` includes `app.api.export.router` with prefix `/api`
   **if the module exists**. So the export session should define `router = APIRouter()` with the path
   `/projects/{pid}/export` (no `/api`). Please add that to section 4 or tell me another convention.
2. Error codes the backend uses beyond the contract's examples: `bad_request`, `not_found`,
   `name_taken`, `class_in_use`, `unknown_model`, `no_classes`, `bad_image_ids`, `bad_class_id`,
   `internal_error`, `method_not_allowed`. Worth listing in section 2 so the UI can rely on them.

3. **`push_image` call shape (for export, #21).** The contract does not fix it, so the worker uses
   this; export please match it or ask for a change:
   * `roboflow_settings()` returns the settings (any object, e.g. a dict) when all 3 names are set,
     or a **list** of missing names.
   * `push_image(settings, image, boxes, class_names, batch_name) -> str` (the Roboflow image id), where
     `image = {"id", "file_name", "path", "width", "height"}`, `boxes = [{"class_name", "bbox": [x, y, w, h]}]`
     (already filtered by status), `class_names` = project class order (#7).
   * The worker loads them lazily in `default_load_pusher()`; tests inject `load_pusher`.
   * If `settings` is a dict, values whose key name contains "key" (and env `ROBOFLOW_API_KEY`) are
     scrubbed from job errors; `api_key=...` in URLs is scrubbed too.

4. **Polygons, bbox-only PATCH that leaves no area (for the integrator, #13).** The contract says
   points are clamped to 0..1, then the bbox is recomputed, but not what happens when nothing is left,
   e.g. `bbox` fully outside the image clamps every point onto one edge. That would store w or h = 0
   and a zero-area polygon, which every other rule forbids, so the backend answers 400 `bad_request`
   and changes nothing. Please add this to the #13 table, or tell me another rule.

## Open questions

1. Should `GET /projects/{pid}/images` reject `limit` > 1000? (currently 400 above 1000.)
2. (closed) `roboflow_push` jobs now run in this worker (B2).
3. The backend work was first built by mistake in the main worktree (`D:/autoannotate/auto-annotate`)
   and then copied here. The untracked copies there (`app/api/`, `app/db/`, `app/worker/`,
   `app/main.py`, `tests/test_api_*.py`, `tests/test_db_schema.py`, `tests/test_worker_status.py`)
   and a smoke-test `data/` folder (project "smoke_test", 1 image) can be deleted by the user.

## Examples mode (#9, #16)

Decisions the contract leaves open, tell me if any is wrong:
1. The cap of 30 is applied after clean-up, so dropped boxes do not use up places.
2. When the detector rejects every class, the job counts as "no class left": done, nothing run or deleted.
3. A box returned for a class that did not run fails the job, like any unknown class.
4. The worker imports `pipeline.prepare_examples` and `Example` only when an examples job runs, and
   without `detector_modes` in autolabel.models every detector has no modes. So merge models with or
   before backend: until then `mode: "examples"` is 400 `unknown_mode` and #16 shows `[]`.

## Teammates + training routes (#37, #38, 2026-11-14)

Built: trained_models `source` / `metrics` / `split` (+ migrate), `store_weights` in app/api/weights.py
(#24 and #38 use it), #24 optional `metrics` part, #25 / #16 fields (`trained`), #37 in app/api/training.py
(`create_app(cuda_available=...)`), #38 `POST /projects/{pid}/weights/import` (AUTOLABEL_IMPORT_DIRS,
default data/inbox, created at startup), tools `start_training` + `import_weights` (19 tools), per-teammate
tokens (app/assistant/users.py, gate in mcp_server.py), scripts/add_teammate.py, mcp_stdio.py `--token`.

Decisions the contract leaves open, tell me if any is wrong:
1. #37 calls export's #23 route function with `{task, epochs, imgsz, base_weights}`, and #24 / #38 call
   export's `read_metrics`. Until export's side is merged, `tests/test_assistant_helpers.export_shims`
   stands in for both (only while they are missing); after the merge the real code runs in the same tests.
2. #38: a relative `path` is 400 `path_not_allowed` (not inside any folder). A metrics file that is too big
   (> 1 MB) or empty is 400 `bad_metrics`; a `metrics` part not named `.json`, or two of them, is 400
   `bad_request`. A sibling `_metrics.json` that resolves outside the folders (symlink) is `path_not_allowed`.
3. `import_weights` checks the metrics file (`bad_metrics`) before asking for confirmation (reading JSON
   runs no code); `no_classes` / `bad_weights` / `task_mismatch` / `class_mismatch` come after `confirm`.
   `details.path` / `metrics_path` are the resolved paths.
4. Loopback Host rule: `127.0.0.1`, `localhost` or `[::1]`, with or without a port.
5. add_teammate's Claude Code line uses the HTTP transport with an `Authorization` header (no launcher
   needed); exact `claude mcp add` flags are a GUESS (not run here).

### Change request (integrator)
`docs/fixtures/assistant/chat_fake.json` `expect_requests.tool_names` still lists the 17 tools of
2026-11-13. The chat sends all 19 (tools.json). `tests/test_assistant_chat.py` now checks the request
against tools.json and that the 17 fixture names appear in that order; please add `start_training` and
`import_weights` after `make_train_package` in the fixture.
(Done in 841afcc.)

## 2026-11-15 E2E gaps

Built: tools `set_review_status` + `add_annotation` (21 tools), `list_images` limit max 50, `web_url` on
Project / Image / Annotation in tool results (BASE = AUTOLABEL_PUBLIC_URL, else MCP Host / chat request base URL,
else http://localhost:8000), absolute `download_url` in tool results, #37 `warnings`, `make_train_package` /
`start_training` risky on warnings (`training.plan_split` -> export's `plan_train_package`, before building),
sidebar SYSTEM line. #12 insert moved to `annotations.insert_annotation` (route + tool share it).

Decisions, tell me if any is wrong:
1. Until export's `plan_train_package` is merged, `plan_split` returns None (no warnings, never risky).
   The tools.json examples make_train_package-0/1, start_training-0/1 and test_api_training's fixture-keys test
   need export's split.json (`val_frames`, `warnings`) and fail until then.
2. `set_review_status`: the `empty` conflict is checked before the >10 confirmation and again inside the
   transaction; one bad id -> nothing changes.
3. `add_annotation` checks the body (bbox/polygon) before the image and class, as #12 does.
