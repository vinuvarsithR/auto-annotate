# export session notes

Build against docs/CONTRACT.md. Only edit files this session owns (CONTRACT.md section 4).

## Done

Part 1 (2026-11-01): `autolabel/export/` pure functions, tests in `tests/test_export_formats.py`
and `tests/test_export_roboflow.py` (SDK mocked, no network).

| Module | What |
|---|---|
| `select.py` | `validate_statuses` (None -> approved default; `rejected`/unknown/empty -> ValueError), `exportable_images` (done/empty), `pending_push_images` (+ no `roboflow_image_id`), `boxes_for_image` (empty image -> no boxes, never `rejected`), `NothingToExport` |
| `formats.py` | `to_coco`, `to_yolo`, `build_zip(fmt, images, classes, annotations, statuses=None, image_bytes=None) -> bytes` |
| `roboflow_push.py` | `roboflow_settings(env=None) -> (settings or None, missing names)`, `batch_name()`, `open_project(settings)`, `push_image(project, image, path, classes, annotations, batch, statuses=None, secrets=None) -> roboflow id`, `PushError` |

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

How backend B2 (worker) should call the push (one SDK project per job):
```python
settings, missing = roboflow_settings()          # missing -> fail with the names only
project = open_project(settings)                 # raises PushError (key-free)
batch = batch_name()                             # job start, UTC
for image in pending_push_images(images):        # Image JSON (+ roboflow_image_id)
    rf_id = push_image(project, image, image_row_path, classes, annotations, batch,
                       statuses=config["statuses"], secrets=[settings.api_key])
    # save rf_id in images.roboflow_image_id, update done/progress
# on PushError: job failed, error = str(exc) (one line, key replaced by ***)
```
`push_image` uploads the file under its original `file_name` (copies it to a temp dir), with a
one-image COCO annotation (`annotation_overwrite=True`), and swallows SDK stdout/stderr.

## Packages needed
- roboflow==1.7.3 (already pinned by integrator). `single_upload(image_path, annotation_path={"name","rawText"}, batch_name, annotation_overwrite)` checked in the installed 1.7.3 source: VERIFIED.
- python-dotenv (already listed).

## Contract change requests
(none)

## Open questions
- An `empty` image is pushed with a COCO file that has an image and zero boxes. Whether Roboflow
  then shows it as "null / background" or as "unannotated" is a GUESS until tried on the real project.
- Part 2 (routes #14, #21 in `app/api/export.py`) waits for backend B1-B3 on main.
