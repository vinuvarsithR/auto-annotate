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

## Packages needed
<!-- package, version, why, VERIFIED or GUESS. The integrator adds them to requirements.txt. -->
None (plain JS, no libraries).

## Contract change requests

## Open questions

## Decided (user, 2026-10-31)
- Re-attach to a running job after a page reload: NOT NOW. Only matters for long real-model
  jobs. Later: add a "list jobs" endpoint (backend + contract), then reconnect the poll on load.
- "Only new images" option for Run detection: NOT NOW. Re-running already keeps every
  human-reviewed box. Revisit when OWLv2 jobs take minutes instead of seconds.
