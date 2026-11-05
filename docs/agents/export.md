# export session notes

Build against docs/CONTRACT.md. Only edit files this session owns (CONTRACT.md section 4).

## Done

Part 1 (2026-11-01): `autolabel/export/` pure functions, tests in `tests/test_export_formats.py`
and `tests/test_export_roboflow.py` (SDK mocked, no network).

| Module | What |
|---|---|
| `select.py` | `validate_statuses` (None -> approved default; `rejected`/unknown/empty -> ValueError), `exportable_images` (done/empty), `boxes_for_image` (empty image -> no boxes, never `rejected`), `NothingToExport` |
| `formats.py` | `to_coco`, `to_yolo`, `build_zip(fmt, images, classes, annotations, statuses=None, image_bytes=None) -> bytes` |
| `roboflow_push.py` | `roboflow_settings() -> RoboflowSettings or list of missing names`, `push_image(settings, image, boxes, class_names, batch_name) -> roboflow id`, `PushError` |

Zip layout (#14):

| Format | Files |
|---|---|
| coco | `annotations.json` (+ `images/<file_name>` with include_images) |
| yolo | `data.yaml`, `classes.txt`, `labels/<stem>.txt` per exported image (empty file = no boxes) (+ `images/<file_name>`) |

- COCO category ids are 1-based in project class order; image/annotation ids are the app ids;
  a mask goes into `segmentation` (RLE), otherwise `[]`.
- YOLO class index = position in the class list (0-based); boxes are clipped to the image,
  boxes fully outside are dropped.
- Two exported images with the same stem: later ones get `<image id>_` in front of the name.

#21 push (changed 2026-11-01): the backend worker does the loop (selection, skipping pushed
images, saving ids). `push_image(settings, image, boxes, class_names, batch_name)` only uploads one
image: `image` = {id, file_name, path, width, height}, `boxes` = [{class_name, bbox}]. It uploads the
file under its original `file_name` (temp copy) with a one-image COCO annotation
(`annotation_overwrite=True`), opens the SDK project once per (workspace, project) (cached), swallows SDK
stdout/stderr, and raises a one-line key-free `PushError`. `RoboflowSettings` hides the key in repr.

Part 2 (2026-11-01): `app/api/export.py` routes #14 and #21, tests `tests/test_export_api.py`.
- #14: loads rows, calls `build_zip`; `include_images` with a missing file -> 404; filename
  `<project>_<format>.zip` (project name reduced to `[A-Za-z0-9._-]`).
- #21: 404 -> 400 statuses (incl. `rejected`) -> 409 settings missing (names only) -> 409 no
  done/empty image; then `db.create_job(..., wake=worker.wake)`; `total` 0 -> done at once.
- One test runs a push job through the real worker + real `push_image` with the SDK faked
  (success and a failing upload whose error has the key in it).

#22 importers (2026-11-03): `autolabel/importers/` (base, safety, coco, yolo, plan), tests
`tests/test_importers_safety.py` + `tests/test_importers_plan.py` (45 tests).
- Choices inside the contract: COCO image covered by nested jsons -> deepest json listing its base
  name wins; YOLO also pairs `images/<sub>/x.jpg` with `labels/<sub>/x.txt` (last `images` folder);
  an unreadable data.yaml falls back to the next config (skipped "could not read class names");
  boxes with no area after clipping are dropped without a `skipped` entry; corrupt zip entries are
  skipped ("cannot unpack"); unreadable images "cannot open image". Duplicate names are exact
  (after `\` -> `/`), not case-insensitive.
- `ImportItem` has no width/height (contract), so the worker re-reads the size when inserting.

Polygons (2026-11-05, from contract-polygons): tests `tests/test_export_polygons.py`,
`tests/test_importers_polygons.py` (fixtures in `docs/fixtures/polygons/`, 1e-6 tolerance).
- `yolo-seg` in `formats.py` (`to_yolo_seg`, 6 decimals), accepted by #14; zip name
  `<project>_yolo-seg.zip`. `to_coco` / `coco_for_image` write pixel `segmentation` + polygon `area`;
  polygon comes before mask. Plain `yolo` writes the polygon's bbox.
- Importers: YOLO polygon rows, COCO `segmentation`, one note per COCO file. Every imported box now
  has a `polygon` key (`None` for a plain box).
- Choices inside the contract: "no area" means shoelace area <= 1e-12 normalized, so float noise on
  collinear points counts as zero. A COCO row with a usable polygon does not need a good `bbox`
  (bbox computed). A COCO row whose box is not stored (bad bbox, or no area left) is not counted in
  the note. `"segmentation": null` is "anything else", so it is counted.
- Real zips, `plan_import` (2026-11-05, VERIFIED, not committed): `v1_yolo.zip.zip` 1649 images, 5,430
  boxes, 5,416 with polygon. `v1_coco.zip` 1649 images, 5,430 boxes, 4,121 with polygon, notes 8 (test) +
  1,280 (train) + 21 (valid) = 1,309; all 1,309 are RLE, none multi-part. So 1,295 boxes have a
  polygon in the YOLO zip but not the COCO zip, because RLE is not decoded (contract).

Polygon test upload (2026-11-05, VERIFIED): SDK, key from `.env`, workspace `vinus-workspace-4ji2t`,
project `test-g9qbh`, **project type `object-detection`**. One 640x360 image (batch
`polygon-contract-test-mixed1`, Roboflow id `MmOF433Jz02aor6G99yq`, left in the project) with one
polygon row and one box row with `segmentation: []`, sent through the real `push_image`. Read back with
`project.image(id)`: 2 rows, polygon `type: "polygon"` with `points` [[64, 72], [192, 72], [128, 216]]
(px, as sent), box `rb` x/y (centre) 400/135, 160x90, as sent. The box row was **not dropped**, so the
4-corner fallback was not added. GUESS: an instance-segmentation project behaves the same; only an
object-detection project was tried.

## Packages needed
- roboflow==1.7.3 (already pinned by integrator). `single_upload(image_path, annotation_path={"name","rawText"}, batch_name, annotation_overwrite)` checked in the installed 1.7.3 source: VERIFIED.
- python-dotenv (already listed).

## Contract change requests
(none)

## Open questions
- An `empty` image is pushed with a COCO file that has an image and zero boxes. Whether Roboflow
  then shows it as "null / background" or as "unannotated" is a GUESS until tried on the real project.
