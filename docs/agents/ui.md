# ui session notes

Build against docs/CONTRACT.md. Only edit files this session owns (CONTRACT.md section 4).

## Done
- 2026-10-31 (NOT committed yet): app/static/ index.html, style.css, api.js (real + ?mock=1 in-memory
  layer over docs/fixtures/api_samples.json), canvas.js (draw/select/move/resize/new box, grey
  placeholder when image missing), app.js (project picker, status filter, paging, progress, A/R,
  Tab, arrows or S/D, 1-9 class, N draw, Esc). Uses only contract endpoints 2,3,5,6,7,11,12,13.
- VERIFIED: node --check on all JS; Node test of mock layer (counts, filter, PATCH status rules,
  POST human box, 404) passes. Server serves page + fixtures (200).
- 2026-10-31 VERIFIED in Chrome (mock mode, static server on repo root, Windows machine), no console
  errors: page + project/status/progress load; drag-move box (bbox x 610->773, status -> edited);
  A accept; 4 sets class biax; R reject; ArrowRight next image; N + drag draws human box (accepted,
  counts update); Esc deselects; Tab selects next box; image list ticks reviewed images.
- 2026-10-31 fixed + VERIFIED in Chrome: banner now floats (position: fixed), canvas top stays put
  when it shows/hides (it auto-hides after 2.5 s, which used to make the canvas jump twice).
  Boxes under 24 screen px on their short side get only 4 corner handles (draw + hit test), so the
  middle stays draggable (small box moved 402->484, size unchanged).
- Committed on branch `ui-review` (app/static/ + this file only).
- 2026-10-31 Setup screen (branch `ui`): "Setup / Review" toggle in the top bar (remembered in
  localStorage). New app/static/setup.js: create project (#1, picker switches to it), upload
  (#4, batches of 10 files via XHR with a byte progress bar, skipped files listed), class table
  (#7/#8: name, description, min_score, accept_score; add/remove rows, "unsaved changes" flag,
  checks names/descriptions/0-1 scores before PUT), Run detection (#16 model dropdown, #9 start,
  #10 polled every 1.5 s with a progress bar, then the review image list + counts refresh).
  Run is blocked while the class table has unsaved changes. Review screen unchanged, apart from
  a `project-opened` event in app.js, loadProjects(preferId), and review shortcuts being off
  while Setup shows. Mock layer: all new calls (409 name_taken, duplicate/not-an-image skips,
  409 class_in_use, 400 no_classes/unknown_model, a fake job that moves 1 image per poll and
  adds the contract's fake-detector boxes with the accept_score rule). Mock uploads use blob:
  URLs, so uploaded images really show on the review canvas.
- #16 shape follows the contract on main (`{"models": [...]}`). This branch's CONTRACT.md does
  not have #16 yet. On 404 the dropdown falls back to `["fake"]` and shows a note.
- VERIFIED in Chrome (mock mode, Windows, static server on repo root), no console errors:
  create project, duplicate name -> banner, 25 PNGs + 1 txt in 3 batches (25 uploaded, txt
  skipped), re-upload -> "duplicate", class save, class_in_use and bad-score messages, job
  polled at ~1.46 s per step to done, 25 auto_accepted + 25 needs_review (accept_score 0.5),
  image list refreshed, uploaded image + boxes visible in Review, A still works, ArrowRight
  ignored on Setup. Real mode with no backend: model list falls back with the note.
- NOT verified against the real backend (uvicorn) yet.
- 2026-10-31 Visual redesign "control room at dusk" (branch `ui`). Same features, endpoint calls,
  shortcuts and ?mock=1; every old element id kept. Colours are CSS variables in style.css
  (base/panel/line/text/muted, one per status); class colours in canvas.js (person, rb, sb, biax +
  an 8-colour muted fallback picked by name hash). Font: IBM Plex Sans 400/500/600 latin woff2 in
  app/static/fonts/ (from npm @fontsource/ibm-plex-sans@5.2.8, OFL.txt alongside).
  VERIFIED: Plex figures are tabular by default ("1111" and "0000" same width).
  Review: thumbnail filmstrip (lazy <img>, amber dot + count / sage dot when done, plain tile when
  the image 404s, as with the fixtures), box panel slides in when a box is selected and pushes the
  image (it stays open, dimmed, while the next image loads), progress = 2px line under the top bar
  with the counts in its tooltip / aria-valuetext, toolbar under the image (title, new-box class,
  draw, show rejected, legend). Setup: 4-step stepper (auto-opens the first unfinished step when
  the project changes), drop zone; dropping or choosing files starts the upload right away (no
  separate Upload button), tiles go dashed -> picture / "Skipped" as each batch of 10 lands.
  Motion (canvas, rAF, user-triggered only, skipped under prefers-reduced-motion; CSS transitions
  off too): accept = sage fill 240ms, reject = fade 200ms, selection glide 50+180ms after
  accept/reject and on Tab, image crossfade 200ms (old image stays until the new one loads, so
  no grey flash).
- VERIFIED in Chrome (mock, Windows, static server on repo root), no console errors: A, R, Tab,
  arrows/D, 1-9 (4 -> biax), drag move (-> edited), N + drag (human box), Esc, ?, Esc closes
  help; effect timings measured (crossfade 202ms, glide 230ms, reject 201ms, accept 240ms);
  reduced motion forced on -> no effects start. Setup: create project -> jumps to Upload;
  25 JPG + 1 txt dropped (DragEvent with DataTransfer) -> 25 tiles + "notes.txt: not an
  image"; re-drop -> 12 duplicates; Run blocked while classes unsaved; save; fake job 25/25;
  25 auto_accepted + 25 needs_review; review shows the uploaded images. 1280px wide checked
  with CSS zoom (window can't shrink below 1707 css px on this screen): toolbar fits with the
  panel open and closed. uvicorn serves index + fonts (font/woff2). pytest 172 passed.
- Not done: real drag from the OS file explorer (only a synthetic drop event was tested);
  the real backend in the browser.
- 2026-10-31 Review tools round (branch `ui`), user-approved plan:
  - Projects start page (new app/static/projects.js, tab "Projects", always opens first): list
    (newest first, images, created date, boxes-reviewed bar), New project (-> Setup, Upload step),
    inline rename (#17), delete with type-the-name confirm (#18), Setup / Review buttons.
    Deleting the open project opens another one, or clears Setup + Review when none is left.
  - Run step compact (no min-height), "Open in review ->" once the job is done / boxes exist.
  - Zoom: Ctrl+wheel or touchpad pinch anywhere on the review screen zooms the image around the
    cursor (100-800% of fit), never the page; plain wheel / drag on empty area / middle button
    pans when zoomed; 0, double-click or the zoom button = fit; every new image starts at fit;
    a programmatically selected box (Tab, after accept) is brought into view.
  - Hold Ctrl: crosshair, drag draws a box (also over existing boxes); N / Draw still work.
  - Delete (Del / Backspace / panel button): human box -> DELETE (#19); model box -> rejected.
    Server without #19 (405) -> rejected instead, with a message.
  - Undo / redo (Ctrl+Z, Ctrl+Shift+Z / Ctrl+Y, toolbar buttons), per image, 50 steps: accept,
    reject, delete, move/resize, class change, draw, repeat previous. Re-created boxes get new ids;
    all steps are re-pointed at them. Undo of a status change needs the #13 change; on an old
    server it says so and keeps the step.
  - Repeat previous (P / button): copies the auto_accepted / accepted / edited boxes of the image
    open before (else the one before in the list) as human boxes; skips same-class IoU >= 0.5;
    scales boxes if image sizes differ; one undo step.
  - Extras the user picked: hold H hides all boxes; legend shows per-class counts on this image
    (rejected not counted) and click hides/shows a class; Adjust popover with display-only
    brightness/contrast (canvas filter, kept across images).
  - Toolbar is now two rows (info: name + legend; tools). Fits at 1280px with the panel open.
- VERIFIED in Chrome (mock, static server on repo root), no console errors: create -> Setup step 2;
  upload 4 generated JPGs, classes, fake job, compact Run step + Open in review; zoom via
  dispatched wheel events (prevented, point under cursor drifts < 1 image px, pinch-size deltas,
  clamp 1-8x, pan, 0 = fit, reset on next image); real Ctrl+click carries ctrlKey; Ctrl-draw
  over an existing box makes a new human box and leaves the old one; Ctrl+Z undid a move exactly;
  draw -> class change -> delete -> 3x undo = exact starting state, 3x redo, server matches;
  model-box delete = rejected, undo -> auto_accepted; repeat previous (1 copied, 2nd press all
  skipped, undo removes); H hide, legend toggle, brightness filter; rename (incl. 409 name_taken),
  delete with confirm (button disabled until the name matches), 409 with a running job, deleting
  the last project -> empty states. Real backend (uvicorn on a scratch data dir): flows work;
  rename/delete project -> 405 message; delete drawn box -> rejected fallback; undo of accept ->
  "can't undo ... contract #13" message. pytest 172 passed.
- GUESS until tried on the user's machine: a real touchpad pinch / real Ctrl+mouse-wheel (Chrome
  sends both as wheel + ctrlKey; only synthetic wheel events were tested), and a real Ctrl+drag
  (the test tool can't hold Ctrl during a drag; tested with synthetic mouse events).

- 2026-11-01 Editor layout (branch `ui`): left icon rail + collapsible Labels panel (classes on this
  image with colour, count, eye toggle; greyed "Unused classes"; Show rejected); canvas with class tags;
  vertical tool rail (Drag D, Box B, Repeat P, Undo, Redo, Mark empty N disabled) with hover tooltips;
  collapsible image strip on the far right (index / total, prev/next, check mark = no box needs review);
  bottom-left bar (zoom - / % / + / Reset, Brightness popover, shortcuts dialog with search).
  Box popup next to the selected box: class field (type to filter, Up/Down, Enter save, Esc cancel),
  Delete, Save. A drawn box stays unsaved (browser only) until Save; Esc drops it. New shortcuts:
  B, D, U (undo approve/reject via #13 explicit status), Ctrl+C / Ctrl+V (one selected box; pasted as
  human boxes, scaled to the image size, one undo step), + / - / 0 / 1 (1 = 100% = 1 image px per screen px).
  Removed: right-side box panel, A/R buttons (keys only), D/S as image nav (Left/Right stay),
  1-9 class keys (1 is now zoom 100%; class via the popup), the "new box class" dropdown (a new box
  starts with the last saved class). Zoom % now shows real pixel scale, not % of fit.
  Fixed: canvas cleared after a resize outside render() (now redrawn on the next frame).
  VERIFIED in Chrome (mock on a static server, real keys/mouse unless noted), no console errors:
  B + drag -> popup focused, "bi" + Enter -> biax human box (1 undo step); Esc drops an unsaved box;
  Esc leaves Box tool; click box, Up, Enter -> class changed (edited); A -> accepted, U -> needs_review,
  R -> rejected, U -> needs_review; Ctrl+C, Right, Ctrl+V -> pasted copy; Delete removes it;
  + 100%, - 80%, 1 100%, 0 fit; N -> "not available" message; ? opens dialog, search "zoom" -> 5 rows
  and no shortcut fired; typing "dar1b" in Reviewer fires nothing; tooltip shows "Box (B)"; strip and
  Labels collapse. Ctrl+drag tested with synthetic events (test tool can't hold Ctrl in a drag).
  Real backend (uvicorn :8011, this worktree's data/app.db): review page loads, boxes drawn, no errors.
  pytest 180 passed. Multi-select is not built (copy works on the one selected box).

## Packages needed
<!-- package, version, why, VERIFIED or GUESS. The integrator adds them to requirements.txt. -->
None (plain JS, no libraries).

## Contract change requests

### 2026-11-01: duplicate uploads by content, not name (for backend + integrator)
Today #4 skips any file whose name already exists in the project (`reason: "duplicate"`), and
`images` has `UNIQUE (project_id, file_name)`. So a different picture that happens to share a
name (e.g. `frame_0001.jpg` from a second camera) is silently lost. The UI can't fix this alone.
Proposal for **#4 POST /projects/{pid}/images**:
- New column `images.sha256 TEXT` (hash of the uploaded bytes; backfilled for old rows on startup).
- Same name AND same sha256 as an image in the project -> skipped, `reason: "duplicate"` (as now).
- Same name, different content -> stored anyway under a free name `stem_2.ext`, `stem_3.ext`, ...
  (keeps the UNIQUE constraint; file names stay unique for COCO/YOLO export). The returned Image
  has the new `file_name`; the response gets `renamed: [{"from": "frame_0001.jpg", "to": "frame_0001_2.jpg"}]`.
- Different name, same content: uploaded as now (not asked for; can be a later option).
The UI will show renamed files in the upload summary. Until then the UI keeps the current behaviour.

### 2026-11-01: mark an image as empty / reviewed (for backend + integrator)
The editor has a "Mark image empty" tool (N). The contract has no image-level state, so the
button is disabled and N only shows a message. Today "done" in the strip = no box needs review,
which can't tell "checked, nothing there" from "never looked at".
Proposal: `images.review_status TEXT NOT NULL DEFAULT 'todo' CHECK (review_status IN ('todo','done','empty'))`,
returned as `review_status` in **Image**, and **#20 PATCH /images/{iid}** `{"review_status": "empty", "reviewer": "dhyan"}`
-> 200 Image. Setting `empty` while the image has boxes that are not rejected -> 409 `conflict`.
#5 gets `?review_status=` as a filter. Export: an `empty` image is exported with no annotations.

### 2026-10-31: project rename/delete, box delete, undo (for backend + integrator)
The UI is already built against these shapes and works with them in `?mock=1`. Against the
real server, rename/delete projects, deleting drawn boxes and undo of status changes stay
unavailable until these are added. The UI does not touch `app/api/`.

New error use: 409 `conflict` (already in the table) for "project has a running job".

**#17 PATCH /projects/{pid}** - rename. Owner: backend.
```json
{"name": "eot_crane_v2"}
```
Response 200 is a Project (same shape as #1, with `counts`). Errors: empty name or bad body ->
400 `bad_request`; name used by another project -> 409 `name_taken`; unknown pid -> 404
`not_found`. Renaming to its own current name is a no-op 200.

**#18 DELETE /projects/{pid}** - delete a project and everything in it. Owner: backend.
No body. Response **204** with an empty body. Removes the project row and (via ON DELETE CASCADE
plus explicit deletes where needed) its images, classes, jobs and annotations, and the folder
`data/uploads/<pid>/` on disk. A job of this project with status `queued` or `running` -> 409
`conflict` and nothing is deleted. Unknown pid -> 404 `not_found`.

**#19 DELETE /annotations/{aid}** - hard delete one box. Owner: backend.
No body. Response **204** with an empty body. Allowed for any box, whatever its source or status
(undoing "draw box" or "repeat previous" must remove the boxes it created). Unknown aid -> 404
`not_found`. The UI itself only hard-deletes human boxes (`source: "human"`); for model boxes
"Delete" still sends `{"status": "rejected"}` so they keep counting toward precision.

**#13 PATCH /annotations/{aid}** - change: `status` may be any of the 5 statuses.
```json
{"status": "needs_review", "reviewer": "dhyan"}
{"status": "auto_accepted", "bbox": [810, 298, 250, 70], "class_id": 4, "reviewer": "dhyan"}
```
Rule: an explicit `status` in the body wins; only when `status` is absent does a bbox / mask /
class change make the box `edited` (as today). Unknown status -> 400 `bad_request`. Undo uses
this to put a box back exactly as it was (e.g. undo "accept" -> `needs_review`, undo "move" ->
the old bbox and the old status in one call). The status rules table in section 1 gets one more
row: "Reviewer undoes a change | the status sent in the request".

## Open questions

## Decided (user, 2026-10-31)
- Re-attach to a running job after a page reload: NOT NOW. Only matters for long real-model
  jobs. Later: add a "list jobs" endpoint (backend + contract), then reconnect the poll on load.
- "Only new images" option for Run detection: NOT NOW. Re-running already keeps every
  human-reviewed box. Revisit when OWLv2 jobs take minutes instead of seconds.
