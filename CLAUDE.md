# CLAUDE.md - project context for Claude Code

Auto-annotation app for a small team, built in phases. Read README.md for
setup and layout. This file holds the working rules, current state and next steps.

## Working rules (from the user, an intern learning as we go)

- Build in phases. Before each phase: a short plan + files to create, then WAIT for OK.
  Stop after each phase and wait for approval.
- Explain each step in plain English. Keep it short; use tables where they help.

## Rules for every session

1. **Start:** `git branch --show-current`, stop if it isn't your branch. `git status`, stop if
   not clean. Then `git merge --no-edit main`.
2. **Ownership.** Never edit files you don't own. Need a change elsewhere: write a contract
   change request in `docs/agents/YOUR_ROLE.md` and stop that part.

   | Role | Owns |
   |---|---|
   | integrator | docs/CONTRACT.md, CLAUDE.md, requirements.txt, docs/fixtures/, merges |
   | backend | app/api/, app/db/, app/worker/, app/main.py + their tests |
   | ui | app/static/ |
   | models | autolabel/models/, autolabel/pipeline.py, one examples-mode method in autolabel/owlv2.py (CONTRACT, Examples mode) |
   | export | autolabel/export/, autolabel/importers/, app/api/export.py + their tests |
3. **Scope:** build only what docs/CONTRACT.md says. No extra features: propose them and wait for OK.
4. **Tests:** evaluation, backend and export code need tests. Run them with
   `D:\autoannotate\auto-annotate\.venv\Scripts\python.exe -m pytest` (worktrees have no .venv).
   All must pass before committing, and before saying anything works.
   Slow tests (`AUTOLABEL_SLOW=1`) need `$env:AUTOLABEL_YOLOE_WEIGHTS="D:\autoannotate\weights\yoloe-26s-seg.pt"`
   (and `AUTOLABEL_YOLO_WEIGHTS` if used).
5. **Commits:** on your own branch only. Never merge into main; the integrator does that
   (`git merge --no-edit YOUR_BRANCH`, pytest, `git push --all`).
6. **App:** run it on a free port, not 8000 or 8001: `uvicorn app.main:app --port YOUR_PORT`.
7. **Roboflow:** the key lives only in `.env` as ROBOFLOW_API_KEY; never print, log or commit
   it. Use the Python SDK, not the Roboflow connector.
8. **Placeholders** in commands look like `YOUR_VERSION`, never `<N>` (bash reads `<` as a redirect).
9. **Report** in at most 5 lines + commit hash. Say VERIFIED vs GUESS (model names, versions,
   VRAM, accuracy).

## Machine (verified 2026-10-30)

Linux, RTX 3060 12 GB (sm_86), driver 595.99, Python 3.10.12,
torch 2.11.0+cu128. Run everything inside `.venv` (`source .venv/bin/activate`).
A second machine (Windows, RTX 5060 Ti 16 GB, sm_120) is not set up yet.

Windows CPU machine (verified 2026-10-31): D:\autoannotate\auto-annotate, NO GPU, PowerShell,
Python 3.14.0, torch 2.15.0+cpu, transformers 5.22.1, fastapi 0.162.0, uvicorn 0.54.0,
Pillow 13.0.0, pytest 9.2.0. Venv: `.venv\Scripts\activate`. No dataset, gold images or
weights here (gitignored); use generated test images (data/test_images/) for app checks.
Used for app work (backend / UI / export), not model runs.

## Phases

| Phase | Scope | Status |
|---|---|---|
| 0 | Setup, GPU check, repo, gold-set folders | Done (Linux) |
| 1 | Standalone OWLv2 script, precision/recall vs gold, threshold sweep | **Parked** (model comparison, see below) |
| 2 | FastAPI backend: upload, class descriptions, job queue, results store | **Built** (endpoints 1-13, 16 models list, #22 bulk import, #4 content dedupe) |
| 3 | Review UI, COCO/YOLO export, Roboflow upload | **Built**: review UI + setup flow (app/static), #14 export zip, #21 Roboflow push. UI import screen not started |
| 4 | SAM masks, second model (Grounding DINO) ensemble, VLM verifier, confidence routing | |
| 5 | Train YOLO / RF-DETR on auto-labels, compare to gold, flag label errors | |
| 6 | Docker, login, docs, demo script | |

Design: FastAPI + SQLite + simple background worker; OWLv2 baseline; favor
recall over precision (humans delete junk faster than they draw missed boxes).

## Parallel sessions (2026-10-31)
Work is split by docs/CONTRACT.md (backend, models, ui, export + integrator on main).
Session notes: docs/agents/*.md. Merged on main (2026-11-03): models (fake + yolo detectors,
pipeline.run, pipeline.drop_overlapping), backend (endpoints 1-13, 16, #21 push jobs, #22 import
route + import jobs, `jobs.result`), export (#14 export zip, #21 push_image, autolabel/importers/),
ui (setup flow, review page, export + push buttons). Run the app: `uvicorn app.main:app --port 8000`,
open http://localhost:8000/. (`/?mock=1` needs a static server on the repo root, e.g.
`python -m http.server`; uvicorn does not serve docs/fixtures/.) Bulk import (#22) has no UI yet:
use http://localhost:8000/docs. Export worktree: D:\autoannotate\aa-export (branch export).
Next:
- backend: wire the #9 overlap rule in app/worker/worker.py (after deleting old model boxes, pass
  the image's non-rejected boxes to `pipeline.drop_overlapping`, store what it returns). Not wired
  yet: nothing calls drop_overlapping (verified 2026-11-03).
- ui: import screen for #22 (zip or folder upload, poll the import job, show `result`).
- 2026-11-13 contract "OWLv2 + SAM 2 polygons" (#27-#30, #9 `masks`, fixtures docs/fixtures/owl_sam/):
  models = merge/hook filter/SAM/polygon, backend = hook points + `masks` + result.hook, ui = hook click + toggle.
  Polygon = single largest part (export/push/import/UI handle one ring only). SAM weights: AUTOLABEL_SAM_WEIGHTS
  (D:\autoannotate\weights\sam2.1_t.pt on this machine).

## Phase 1 state

Code: `autolabel/` (owlv2.py, evaluate.py, gold.py, coco.py), `scripts/run_owlv2.py`,
`scripts/fetch_gold_from_roboflow.py`, thresholds.py (step C done), scripts/compare_runs.py (step A done), sequences.py, dedup.py, split.py, gold examples, mAP, merge, gdino, report. 107 tests pass.

Gold set: `gold_sets/eot_crane_cam0/` - PROVISIONAL (copied from Roboflow labels,
not hand-checked), 50 images: person 102, rb 19, sb 16, biax 20 boxes.
Source: workspace `vinus-workspace-4ji2t`, project `eot_crane-yduh1-vhu5p-ohmpg`.
Overhead crane camera looking straight down. Classes (confirmed by user with examples):
- rb: long narrow beige slab hanging along the crane cables
- sb: very long thin yellow beam held horizontally under the hook
- biax: SHORT wide yellow/orange crossbar right under the hook
- person: workers in hard hats seen from above
Image filenames name the crane job (e.g. `biax__cam0_...`), NOT the labeled object.

Run 1 results (model google/owlv2-base-patch16-ensemble, text prompts in classes.yaml):
- Speed 2.9 img/s, peak VRAM 0.76 GB (verified).
- At 0.20: person R 0.21, rb R 0.00, biax R 0.00, sb R 1.00 but P 0.04.
- Class-ignored recall 0.93 at 0.05 vs 0.69 plain: boxes found, wrong class.
- Confusion check at 0.10: rb -> sb 18/19, biax -> sb 16/20, sb -> sb 16/16,
  person -> person 75/102. The "yellow beam" text prompt claims all three loads.
- A single global 0.20 threshold is too high for this data.

Step A results (person prompt only; files in experiments/person_prompts/, runs/A1-A3):
| prompt | R@0.05 | P@0.05 | R@0.10 | P@0.10 |
| baseline "a worker wearing a hard hat, seen from above" | 0.90 | 0.09 | 0.74 | 0.26 |
| A1 "a person" | 0.64 | 0.49 | 0.47 | 0.77 |
| A2 "a hard hat" | 0.62 | 0.13 | 0.23 | 0.13 |
| A3 "a construction worker" | 0.84 | 0.23 | 0.61 | 0.56 |
rb/sb/biax unchanged across variants. Baseline kept for now (best recall); user to decide.

## Next steps (approved by user)

A. Try 2-3 alternative `person` descriptions ("a person", "a hard hat", ...),
   re-run, compare per-class recall. Report a table.
B. Image-guided queries for rb, sb, biax: OWLv2 supports example-image queries
   (`query_images` in Owlv2Processor, `model.image_guided_detection`,
   `processor.post_process_image_guided_detection` - present in transformers; INSTALLED version is 5.20.2, verified 2026-10-30).
   Let classes.yaml give example crops per class; keep text prompts for others.
   Compare against the text-prompt run with the same gold set. Add tests where
   logic is testable without a GPU. Expected improvement is a GUESS until measured.
   Note image-guided post-processing applies its own NMS (nms_threshold).
C. Per-class thresholds chosen from the sweep (e.g. highest threshold keeping
   recall >= a target), reported per class. Tests for the selection logic.

Then show results and wait for approval before Phase 2.

Step B findings so far (verified 2026-10-30, transformers 5.20.2 source):
- post_process_image_guided_detection rescales scores PER IMAGE (best box = 1.0
  even with no object). Unusable for thresholds; use sigmoid(logits) + own NMS.
- Whole Roboflow export is local: data/roboflow/eot_crane-yduh1-vhu5p-ohmpg-v1 (1649 imgs).
- Only 8 video sequences (~1 frame/s, GUESS from start times). Gold uses 7 of 8.
  sb boxes exist ONLY in biax__..18-48-25; biax ONLY in rb_load__..18-24-11 / 18-34-11;
  sb_evener sequence has no sb box. Sequence-level separation is impossible for sb/biax.
- autolabel/sequences.py parses sequence + frame from (Roboflow-renamed) file names. 10 tests.
- User chose: get MORE FOOTAGE (not far-frames).
- Roboflow access rule (user, 2026-10-30): ONLY the Python SDK with ROBOFLOW_API_KEY from .env
  (vinus key). Do NOT use the Roboflow MCP connector (signed in to another workspace) and do
  not use any data from workspace adil-muhammed-ashraf.
- SDK survey of vinus-workspace-4ji2t (read-only, roboflow 1.7.3, verified 2026-10-30):
  2 projects. eot_crane-yduh1-vhu5p-ohmpg: 1649 images, 8 sequences, 0 unannotated,
  1 batch + 1 complete job (fork), 1 version. sb ONLY in biax__..18-48-25 (258 imgs);
  biax only in rb_load__..18-34-11 (235) and 18-24-11 (43). Description search for a long
  yellow beam found nothing outside these 8 sequences. blade_rework-7u6si: different use case,
  no sb/biax labels; its 2360 unannotated images were NOT reachable by RoboQL search.
- SDK gotchas: Workspace.list_projects() prints and returns None (use .project_list);
  RoboQL 'NOT dataset:' did not return unannotated images in blade_rework.
- sequences.py also parses `_tNNNN` (added for other-workspace names; harmless, user may revert).
  It does NOT parse blade_rework's `_fNNNNN` names (not needed for crane).
- User confirmed 2026-10-30: NO other crane images exist; the v1 export (1649 imgs, 8 videos) is
  the whole dataset (~/Downloads/eot_crane.v1i.coco.zip is byte-identical to it).

## Model comparison task (approved plan, 2026-10-30) - PARKED 2026-10-31
User: "only building the pipeline" for now; models get plugged in later. Parked, not cancelled.
Resume point is below (YOLO paused at epoch 11/50 on the Linux machine, OWLv2/GDINO runs and
compare_models.py not run). Don't propose resuming it until the user asks.

Compare OWLv2 text, OWLv2 image-guided (= step B), Grounding DINO, YOLO on a new test set.
Split BY TIME inside each video: [val 10%][gap][train][gap][test], gap = max(5%, 15 frames),
unpicked test-window frames unused, train/val look-alikes of test dropped (dHash 256-bit).
Per-class thresholds picked on VAL (target recall 0.90), applied to test. mAP@0.5 too.
sb+biax: report keep-both and keep-higher for B. HTML report with toggles, licence (Ultralytics
AGPL-3.0) and provisional-label notes. Checkpoints: (1) split table, (2) example crops, (3) results.
Code so far: autolabel/dedup.py, autolabel/split.py, sample_coco(accept=...), scripts/make_split.py.
dHash measured: neighbour frames median 47/256 bits, random different-video pairs median 110.
Step 1 DONE (user chose test 30%, look-alike <=16 bits, test gap 5 frames): test50 has 47 imgs
(biax 21, person 112, rb 28, sb 5 boxes), train 800, val 126, 22 train look-alikes dropped. Leak check OK.
User is on a TIME CONSTRAINT: wants results asap.
YOLO: ultralytics 8.4.193 (latest on PyPI 2026-10-30, AGPL-3.0, did not change torch), model yolo26s
(in its asset list), imgsz 1280, 50 epochs, patience 15. scripts/train_yolo.py, log runs/yolo_train.log,
weights runs/yolo/eot_crane_v1_yolo26s/weights/best.pt. coco_to_yolo() in coco.py with tests.
Grounding DINO ids VERIFIED on HF: IDEA-Research/grounding-dino-tiny and -base.
Step 2 (checkpoint): examples in experiments/image_queries/B1.yaml, sheet runs/B/examples_sheet.jpg.
All 3 biax examples come from rb_load 18-34-11 (18-24-11 biax frames fall in its test/gap window).
Crops approved by user. Code written (tests pass, 107): evaluate.average_precision/map50, merge.keep_higher,
owlv2 image-guided (query_embedding + detect_image_guided, own sigmoid scoring), gdino.py (base model,
own token->class scoring), report.py, thresholds.apply_thresholds, scripts run_models.py + compare_models.py.
YOLO training restarted detached (first run died at epoch 1 when the session ended). PAUSED BY USER at
epoch 11/50 with SIGSTOP on python pid 3863 + its worker children (still holds ~11 GB GPU).
Resume: kill -CONT 3863 and its children (pgrep -P 3863). Weights so far: runs/yolo/eot_crane_v1_yolo26s/weights/{best,last}.pt
(val mAP50 0.67 after epoch 10). Not run yet: owl_text/owl_image/gdino on GPU (GPU was full), compare_models.py.
- (sb footage question closed: no more footage exists.)
Also: the provisional gold labels should get a quick visual check; one long
yellow beam was seen labeled `rb` instead of `sb` in Roboflow.
