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
