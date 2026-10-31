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

## Packages needed
<!-- package, version, why, VERIFIED or GUESS. The integrator adds them to requirements.txt. -->

## Contract change requests

## Open questions
