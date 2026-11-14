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

- 2026-11-03: **`yolo` detector** (`autolabel/models/yolo.py`, `YoloDetector`), same interface as `fake`/`owlv2`.
  Weights = option `weights`, else env `AUTOLABEL_YOLO_WEIGHTS` (missing -> `ValueError` naming the variable).
  Classes matched **by exact name** (`model.names` -> `ClassSpec.name`), never by index; model classes the
  project lacks are not even asked for (`classes=` filter); project classes the model lacks get no boxes.
  `description` is unused (no prompts). Floor per class = `min_score`, else 0.05 (same recall-first default as
  owlv2, not tuned for YOLO). Options: `weights`, `device` ("cpu"), `min_score`, `imgsz` (None = the model's own).
  Off by default: enable with `AUTOLABEL_DETECTORS=yolo` (or `owlv2,yolo`). One instance per process.
  Tests: `tests/test_models_yolo.py` (14 mocked + 1 slow with stock `yolo11n.pt` on ultralytics' bus.jpg).
  VERIFIED 2026-11-03 Windows CPU: yolo11n 0.038 s/image (810x1080, default imgsz 640). The Colab best.pt
  will be slower if it is a bigger model or trained at imgsz 1280 (GUESS until measured).
- 2026-11-03: **yolo image size.** `imgsz` = option, else env `AUTOLABEL_YOLO_IMGSZ` (positive int, else
  `ValueError`), else the checkpoint's `train_args["imgsz"]` (then `model.overrides["imgsz"]`), else 640.
  Always passed to `predict`. Tests: 8 new mocked tests (precedence, bad env values); the slow test checks the
  real yolo11n.pt reports 640. `D:\autoannotate\weights\yolo11s_960_v2.pt` was NOT on this machine, so its
  speed is not measured. Stand-in (same architecture and size): stock yolo11s.pt at imgsz 960 on a 1280x720
  frame, Windows CPU = 0.133 s/image (0.072 s at 640). VERIFIED for the stand-in only; the real file should be
  close (GUESS).

## Examples mode (CONTRACT #9, 2026-11-05)
Built: `autolabel/models/yoloe.py`; `prepare_examples` on `fake` and `owlv2`; `Example`, `ExampleRun`,
`Prediction.polygon`; `detector_modes`; `pipeline.prepare_examples`, `run(example_run=...)`.
Real check on the spike's 40 frames (VERIFIED, CPU, thresholds tuned on those frames): avg F1 0.415,
0.21 s/img at the default floor 0.05 including polygons; `prepare_examples` 6.4 s for 19 images.

- 2026-11-06: **owlv2 examples fix** (CONTRACT, `owlv2` examples-mode rules). New `Owlv2Detector.image_guided_logits`
  in `autolabel/owlv2.py` (raw logits + boxes); `detect_image_guided` is no longer used by examples mode but is
  unchanged (scripts/run_models.py still calls it). Score = `sigmoid(logit - class_cutoff)`, cutoff = median example
  logit, floors on that score, top 5 per class per image after NMS and clipping. NMS and the cap order by
  `logit - cutoff`, because the sigmoid rounds to 1.0 far above the cutoff. Cutoffs add one model pass per example image.
  Real check (VERIFIED, Windows CPU, cutoffs from the examples, nothing tuned): 40 spike frames, floor 0.05: 20.0
  boxes/img (629.5 without the cap), macro-F1 0.173, 4.8 s/img; prepare 268 s for 19 images, 37 examples. Same numbers
  as calib2 on the saved logits. Cap 5 costs recall: sb 0.90 -> 0.60, biax 0.88 -> 0.38, person 0.99 -> 0.87.
- 2026-11-10: **owlv2 examples = spike method, rule C** (CONTRACT 6845950). Replaced the prototype mean, class_cutoff,
  calibrated sigmoid and top-5 cap. `autolabel/owlv2.py`: `box_features` (normalized class embeddings, boxes,
  objectness) replaces `image_guided_logits`. `ExampleRun.skip_reasons` added to base.py. On the user's instruction the
  `pos` row was written straight into CONTRACT.md (was a placeholder), with the `bg` details pinned: overlap = IoU >= 0.3,
  objectness >= 0.01 counts, no bg value -> threshold = pos. One model pass per example image.
  40-frame benchmark, 5 examples/class, Windows CPU, nothing tuned (VERIFIED): macro-F1 0.546, 3.6 boxes/img,
  4.6 s/img, prepare 88 s (spike C was 0.559 / 4.0). Person is lower (R 0.61 -> 0.37) because its example images
  hold more persons than the 1 example box, and the contract counts them as background: bg rises above pos, so the
  threshold becomes pos (0.869, spike 0.846).
- 2026-11-10: **owlv2 bg uses `Example.labelled`** (CONTRACT def255c). `Example.labelled` added to base.py; bg excludes
  the union of `bbox` and `labelled` per image. Filtering (non-rejected, any source/status, cap ignored) is the
  worker's job; models only trusts the list. Benchmark, 5 examples (1 box per image) + labelled = the other gold boxes
  of the class there (person 15, biax 2), Windows CPU (VERIFIED): macro-F1 0.586, 4.0 boxes/img, person P 0.91 R 0.56,
  4.8 s/img, prepare 134 s. Person threshold 0.869 -> 0.850. With every person box in those images given as examples: macro-F1 0.564, 1.8 boxes/img.

## Packages needed
<!-- package, version, why, VERIFIED or GUESS. The integrator adds them to requirements.txt. -->
- None new for `fake`. `owlv2` needs torch, torchvision (used by `autolabel/owlv2.py` for NMS) and transformers.
  VERIFIED installed in the shared Windows venv 2026-11-02: torch 2.15.0+cpu, torchvision 0.29.2+cpu,
  transformers 5.22.1 (Python 3.14.0); nothing had to be installed.
- **`ultralytics` (for `yolo`), please add to requirements.txt:** `ultralytics==8.4.199` (AGPL-3.0).
  VERIFIED 2026-11-03: installs and imports on Python 3.14.0 in the shared Windows venv; it did NOT change torch
  (2.15.0+cpu) or torchvision. It pulled in opencv-python 5.0.0.93, polars 2.1.0, psutil 7.2.2,
  nvidia-ml-py, ultralytics-thop 2.2.4, ultralytics-platform 0.1.102. It is imported only inside
  `YoloDetector.__init__`, so the app runs without it unless `yolo` is enabled.
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

4. **Choose a detector per class (needed for the demo: person from owlv2 or yolo, rb/sb/biax from yolo).**
   Today a job runs exactly ONE detector: `pipeline.run(..., detector=name)` calls `get_detector(name)`
   for every class; `AUTOLABEL_DETECTORS` only decides which names are *offered*, it never combines them.
   `"pipeline"` is still unregistered. So per-class choice needs one of these (NOT built, waiting for OK):
   - (a) **`ClassSpec.detector: str | None = None`** + a `classes.detector TEXT` column (NULL = the job's
     model) settable in #7/#8. `pipeline.run` groups classes by detector, calls each once per image and
     concatenates. Per-project, editable in the UI later. Touches base.py, backend DB/API, CONTRACT.
   - (b) **Register `"pipeline"` in models only**, with routing from env, e.g.
     `AUTOLABEL_PIPELINE=person:owlv2,rb:yolo,sb:yolo,biax:yolo` (unlisted classes -> a default detector).
     No interface change, but the routing is per server, not per project, and the backend still sees one model.
   - Recommendation: (b) for the demo now (smallest change, models-only), (a) later if projects need different
     routings. Either way the contract must say what `"pipeline"` means (CONTRACT #9/#16).
   - Note: both detectors returning `person` would give duplicate boxes; "owlv2 OR yolo" means routing picks one.
     Merging both (`merge.keep_higher`) is a separate choice.
5. **CONTRACT section 3/#16:** `"yolo"` exists when `AUTOLABEL_DETECTORS` contains `yolo` and
   `AUTOLABEL_YOLO_WEIGHTS` points at the weights; add `'yolo'` to the `annotations.source` comment.

6. **CONTRACT #9, `yoloe` prototype (integrator):** please say "one embedding per image per class: all
   of the class's boxes in an image form one prompt". Built this way, because the wording "mean of its
   example embeddings" also fits one embedding per box, and that scored much worse on the spike's 40 frames:
   avg F1 0.33 against 0.42, person 0.54 against 0.62, biax 0.01 against 0.27 (VERIFIED, small sample).
   `used` still counts boxes, so `examples_by_class` means the same.
7. **Weights (integrator / README):** `D:\autoannotate\weights\yoloe-26s-seg.pt` does not exist yet. Tests
   and the real check used the spike's copy through `AUTOLABEL_YOLOE_WEIGHTS`.

8. **Done 2026-11-05:** once ultralytics is imported, a file Pillow cannot identify runs `pip install pi-heif`
   in the live environment. `yolo.disable_autoinstall()` turns this off before `yolo` or `yoloe` load a model.
   The `pi_heif` a test run put in the shared .venv was removed the same day.

## Open questions
- Should `get_detector` key its cache on `(name, options)` instead of `name` only? With real models, two jobs with
  different options (e.g. a different checkpoint) would currently share the first instance.

## 2026-11-13: OWLv2 + SAM 2 polygons (#9 masks, steps 1-4)
Built: `PostProcess` (base.py), `autolabel/models/postprocess.py` (merge_contained, hook_filter),
`autolabel/models/sam.py` (Segmenter, Sam2Segmenter, get_segmenter, FakeSegmenter, clean_mask, largest_polygon,
add_polygons), `pipeline.run(post=, segmenter=)`. Tests: test_models_postprocess.py (all owl_sam fixture cases),
test_models_sam.py, test_pipeline_post.py (no weights).
Read-only check (D:\autoannotate\spikes\owl_sam_check, real OWLv2 + SAM 2.1 tiny, 40 spike test frames, VERIFIED):
with one-load rule box F1 0.72 / mask F1 0.74; without it 0.71 / 0.72 (= spike v2 largest-part result).
The rule only changes biax (0.40 -> 0.53 mask F1, no true positive lost). 7.6 s/img CPU (OWLv2 5.4 + SAM 1.8),
peak RAM 1.94 GB in one process.

## 2026-11-14 - contract-owl-sam-close: 25 px closing in step 3
- `sam.CLOSE_PX = 25`; `clean_mask(..., close_px=CLOSE_PX)`: pick -> 5 px opening -> 25 px closing (MORPH_ELLIPSE) -> clip; 0 = off.
- Tests (fake masks): closing joins two parts 20 px apart, `close_px=0` leaves two, closing never adds pixels outside the box,
  CLOSE_PX value; the two-part test now uses parts 27 px apart (more than the closing joins).
- owl_sam_check rerun (built pipeline, 40 test frames, real OWLv2 + SAM 2.1 tiny, VERIFIED): with one-load rule box F1 0.76 /
  mask F1 0.78 (was 0.72 / 0.74); rb 0.92 / 0.92 (median IoU 0.78 -> 0.83), sb 0.90 / 1.00 (was 0.80 / 0.80), biax 0.67 / 0.67,
  person 0.69 / 0.71. Same as the gap spike's close25 row.
