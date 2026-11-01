"""
select.py - which images and boxes get exported / pushed (CONTRACT section 1,
"Export / push rules").

Everything works on the API JSON shapes (Image, Class, Annotation dicts).
"""

from __future__ import annotations

# Image review statuses that are exported. `todo` images are left out.
EXPORT_REVIEW_STATUSES = ("done", "empty")

# The 5 box statuses of #13, and the default "approved" subset.
ALL_STATUSES = ("auto_accepted", "needs_review", "accepted", "rejected", "edited")
APPROVED_STATUSES = ("auto_accepted", "accepted", "edited")


class NothingToExport(Exception):
    """No image has review_status done/empty (the route answers 409 conflict)."""


def validate_statuses(statuses: list[str] | None) -> list[str]:
    """
    Return the box statuses to export. None means the approved default.
    Raises ValueError for an empty list, `rejected`, an unknown status or a non-list
    (the route answers 400 bad_request).
    """
    if statuses is None:
        return list(APPROVED_STATUSES)
    if not isinstance(statuses, (list, tuple)) or not statuses:
        raise ValueError("statuses must be a non-empty list")
    for s in statuses:
        if s == "rejected":
            raise ValueError("rejected boxes are never exported")
        if s not in ALL_STATUSES:
            raise ValueError(f"unknown status: {s!r}")
    # keep the caller's order, drop repeats
    return list(dict.fromkeys(statuses))


def exportable_images(images: list[dict]) -> list[dict]:
    """Images with review_status done or empty, in the given order."""
    return [im for im in images if im.get("review_status") in EXPORT_REVIEW_STATUSES]


def pending_push_images(images: list[dict]) -> list[dict]:
    """
    Exportable images that were not pushed to Roboflow yet (#21 `total`).
    An image counts as pushed when its dict has a non-empty `roboflow_image_id`.
    """
    return [im for im in exportable_images(images) if not im.get("roboflow_image_id")]


def boxes_for_image(image: dict, annotations: list[dict], statuses: list[str]) -> list[dict]:
    """
    Boxes of one image whose status is in `statuses`. An `empty` image always gets
    no boxes. `rejected` is never returned, whatever `statuses` says.
    """
    if image.get("review_status") == "empty":
        return []
    keep = set(statuses) - {"rejected"}
    return [a for a in annotations if a["image_id"] == image["id"] and a["status"] in keep]
