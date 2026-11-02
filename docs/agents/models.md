# models session notes

Build against docs/CONTRACT.md. Only edit files this session owns (CONTRACT.md section 4).

## Done
- `autolabel/models/base.py`: `ClassSpec`, `Prediction(Box)`, `Detector` (Protocol), copied word for word from CONTRACT section 3.
- `autolabel/models/__init__.py`: `list_detectors()` returns `["fake"]`; `get_detector(name, **options)` builds each
  detector once and reuses it (options only count on the first call); an unknown name raises `ValueError`.
  Detector modules are imported inside their factories, so `import autolabel.models` never loads torch.
- `autolabel/models/fake.py`: `FakeDetector` (`name = "fake"`), no GPU and no weights. It returns 2 boxes per image
  (first class 0.9 at `[0.1W, 0.1H, 0.3W, 0.3H]`, second class 0.3 at `[0.5W, 0.5H, 0.3W, 0.3H]`), `source="fake"`,
  `mask=None`. It honours `min_score`; with fewer than 2 classes it returns fewer boxes.
- `autolabel/pipeline.py`: `run()` as in the contract. It is a lazy generator that opens each image with PIL, converts it
  to RGB, and calls `on_progress(done, total)` before yielding each image.
  **`detector="pipeline"` (the contract default) raises `ValueError("... not available yet")` until the real
  pipeline exists.** Backend tests should pass `detector="fake"`.
- Tests: `tests/test_models_fake.py` (11), `tests/test_pipeline_run.py` (7). Full suite: 125 passed (2026-10-31, Windows).

- 2026-11-02: **`owlv2` detector** (`autolabel/models/owlv2.py`, `Owlv2TextDetector`), same interface as `fake`.
  Wraps the read-only Phase 1 `autolabel/owlv2.py` (model `google/owlv2-base-patch16-ensemble`).
  Prompt per class = `ClassSpec.description`, or `name` if the description is empty. Score floor per class =
  `min_score`, else 0.05 (`DEFAULT_MIN_SCORE`): the lowest Phase 1 sweep threshold (CLAUDE.md Run 1 / Step A,
  CONTRACT examples). Phase 1 never chose final per-class thresholds (model comparison parked), so 0.05 is a
  recall-first default, not a tuned value. `source="owlv2"`.
  **Off by default:** `list_detectors()` is `["fake"]` unless `AUTOLABEL_DETECTORS=owlv2` (comma list, read on
  every call). Weights download once to the Hugging Face cache; `get_detector` keeps one instance per process.
  Options: `model_id`, `device` (auto/cpu/cuda), `nms_iou` (0.5), `min_score` (0.05).
  Tests: `tests/test_models_owlv2.py` (11 mocked + 1 slow, skipped unless `AUTOLABEL_SLOW=1`;
  `AUTOLABEL_SLOW_IMAGE=path` for a real frame). VERIFIED 2026-11-02 Windows CPU: 5.7 s/image on a 1280x720
  crane frame (rb_load 18-34-11 #00378), 58 boxes at 0.05.

## Packages needed
<!-- package, version, why, VERIFIED or GUESS. The integrator adds them to requirements.txt. -->
- None new for `fake`. `owlv2` needs torch, torchvision (used by `autolabel/owlv2.py` for NMS) and transformers.
  VERIFIED installed in the shared Windows venv 2026-11-02: torch 2.15.0+cpu, torchvision 0.29.2+cpu,
  transformers 5.22.1 (Python 3.14.0); nothing had to be installed.
- VERIFIED on the Windows machine (2026-10-31): Python 3.14.0, Pillow 13.0.0, pytest 9.2.0.
  The Linux machine has Python 3.10.12. The code uses `X | None` only under `from __future__ import annotations`,
  so it should run on 3.10 too (GUESS, not run there yet).

## Contract change requests
1. **`Prediction` with a mask is not hashable.** `Box` is `@dataclass(frozen=True)`, so Python adds `__hash__`.
   `Prediction.mask` is a `dict`, so `hash(prediction)` raises `TypeError` whenever a mask is set
   (VERIFIED by `test_prediction_with_mask_is_not_hashable`). Anything that puts predictions in a set or uses them
   as dict keys will break once SAM masks arrive (Phase 4). Options: (a) declare
   `mask: dict | None = field(default=None, hash=False, compare=False)`, or (b) store the mask as a JSON string.
   Recommendation: (a). Not changed, because base.py must match the contract exactly.

2. **requirements.txt (integrator):** transformers is already listed. torch is installed by hand (README step 4,
   cu128 index). Please add `torchvision` to that install line/comment (needed by `autolabel/owlv2.py`) and a CPU
   alternative for the Windows machine: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.
3. **CONTRACT section 3/#16 (integrator):** note that `"owlv2"` exists and appears in `list_detectors()` /
   `GET /api/models` only when the server runs with `AUTOLABEL_DETECTORS=owlv2`.

## Open questions
- Should `get_detector` key its cache on `(name, options)` instead of `name` only? With real models, two jobs with
  different options (e.g. a different checkpoint) would currently share the first instance.
