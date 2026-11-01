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
(`annotation_overwrite=True`), opens the SDK project once per settings (cached), swallows SDK
stdout/stderr, and raises a one-line key-free `PushError`. `RoboflowSettings` hides the key in repr.

## Packages needed
- roboflow==1.7.3 (already pinned by integrator). `single_upload(image_path, annotation_path={"name","rawText"}, batch_name, annotation_overwrite)` checked in the installed 1.7.3 source: VERIFIED.
- python-dotenv (already listed).

## Contract change requests
(none)

## Open questions
- An `empty` image is pushed with a COCO file that has an image and zero boxes. Whether Roboflow
  then shows it as "null / background" or as "unannotated" is a GUESS until tried on the real project.
- Part 2 (routes #14, #21 in `app/api/export.py`) waits for backend B1-B3 on main.
