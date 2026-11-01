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

## Packages needed
- roboflow==1.7.3 (already pinned by integrator). `single_upload(image_path, annotation_path={"name","rawText"}, batch_name, annotation_overwrite)` checked in the installed 1.7.3 source: VERIFIED.
- python-dotenv (already listed).

## Contract change requests
(none)

## Open questions
- An `empty` image is pushed with a COCO file that has an image and zero boxes. Whether Roboflow
  then shows it as "null / background" or as "unannotated" is a GUESS until tried on the real project.
