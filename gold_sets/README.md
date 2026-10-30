# Gold sets

A gold set is a small batch of images with carefully checked labels. It's the
answer key the auto-labeler is scored against, so accuracy matters more than size.

## One gold set per (classes + camera view)

Projects can share a gold set only if they have the **same classes** AND
**similar images** (same camera view, lighting, background). Our projects use
different camera views, so most camera views will need their own gold set.

Name each gold set after what's in it, not after a project
(e.g. `root_scissor_lift_cam`, not `lm_yellow_scissor_lift_ft`).

## Which project uses which gold set

Pulled from Roboflow on 2026-10-30. Fill in the last two columns as a team.

| Roboflow project | Class group | Camera view | Gold set |
|---|---|---|---|
| LM Combined | Blade root | _fill in_ | _fill in_ |
| lm_cleaned_up_set | Blade root | _fill in_ | _fill in_ |
| lm_cleaned_up_set 2 | Blade root | _fill in_ | _fill in_ |
| lm_root_monitoring_sept_24 | Blade root | _fill in_ | _fill in_ |
| blade_rework | Blade root (+ `stair`) | _fill in_ | _fill in_ |
| v19_fail_rework | Blade root | _fill in_ | _fill in_ |
| v3_field_rework | Blade root | _fill in_ | _fill in_ |
| lm_label_audit | Blade root | _fill in_ | _fill in_ |
| root_wagon_c | Blade root | _fill in_ | _fill in_ |
| lm_yellow_scissor_lift_ft | Blade root | _fill in_ | _fill in_ |
| eot_crane | EOT crane | _fill in_ | _fill in_ |
| eot_crane_oct17 | EOT crane | _fill in_ | _fill in_ |
| eot_crane_box_rework | EOT crane | _fill in_ | _fill in_ |
| eot_crane_sep15_biax | EOT crane | _fill in_ | _fill in_ |

## Canonical class names

Roboflow uses several spellings for the same class. Gold sets use only the
names in the first column.

| Canonical name | Spellings seen in Roboflow | Class group |
|---|---|---|
| `person` | `person`, `Person`, `a_person` | Both |
| `open_root` | `open_root` | Blade root |
| `closed_root` | `closed_root` | Blade root |
| `closed_root_opening` | `closed_root_opening` | Blade root |
| `stair` | `stair` | Blade root |
| `root_lamination` | `root_lamination` | EOT crane |
| `aerolift` | `aerolift` | EOT crane |
| `biax` | `biax` | EOT crane |
| `rb` | `rb` | EOT crane |
| `sb` | `sb` | EOT crane |

Ignored (junk or empty classes): `1`, `2`, `3`, `4`, `7`, `d`, `D`,
`eot-crane`, `lm-root-monitoring-sept-24`, `lm-yellow-scissor-lift-ft`.

## How to make a new gold set

1. Copy `_template/` to a new folder named after the content.
2. Pick **50–100 images** from that camera view. Include hard cases
   (crowded, dark, partly hidden), not just easy ones.
3. Aim for at least ~50 boxes per class (a rough guess, not a standard).
4. Start from existing Roboflow labels, then **re-check every box by hand**.
   Some existing labels are known to be wrong (see the `audit` and `rework`
   projects), and the gold set must be correct.
5. Export as **COCO JSON**, rename to `annotations.json`, put the images in
   `images/`. The `file_name` values must match the image filenames exactly.
6. Fill in `classes.yaml` using canonical names only.
7. Add the gold set to the table above.

## Labeling rules (one rulebook for every gold set)

Agree on these before anyone checks labels:

| Question | Our rule |
|---|---|
| Box partly hidden objects? (e.g. 30% visible) | _fill in_ |
| Box objects cut off at the image edge? | _fill in_ |
| Tiny objects below what size are skipped? | _fill in_ |
| A group of overlapping people: one box or one each? | _fill in_ |
| When is a root `closed_root_opening` vs. `closed_root`? | _fill in_ |

## COCO format in 30 seconds

```json
{
  "images":      [{"id": 1, "file_name": "img_001.jpg", "width": 1920, "height": 1080}],
  "categories":  [{"id": 1, "name": "person"}],
  "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                   "bbox": [100, 200, 50, 80], "area": 4000, "iscrowd": 0}]
}
```

`bbox` is `[x, y, width, height]` in pixels, measured from the **top-left**
corner of the image.
