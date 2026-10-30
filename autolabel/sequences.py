"""
sequences.py - work out which video sequence an image came from.

Our frames are cut from videos. A file name like

    rb_load__cam0_2026-09-29_18-24-11_00568.jpg

means sequence 'rb_load__cam0_2026-09-29_18-24-11', frame 568. Roboflow
exports rename files, e.g.

    rb_load__cam0_2026-09-29_18-24-11_00568_jpg.rf.<hash>.jpg

Some projects number frames as '_t0036' instead of '_00036'.

We strip the '.rf.<hash>' part and the '_jpg' that replaced the old dot
before splitting off the frame number.

Neighbouring frames look almost the same, so anything that keeps data apart
(test vs train, example crops vs gold) should keep whole SEQUENCES apart.
"""

from __future__ import annotations

import re
from pathlib import PurePath

_ROBOFLOW_SUFFIX = re.compile(r"\.rf\.[0-9a-fA-F]+$")
_OLD_EXT = re.compile(r"_(jpe?g|png|bmp|webp)$", re.IGNORECASE)
_FRAME = re.compile(r"^(?P<seq>.+)_t?(?P<frame>\d+)$")  # some exports write _t0036


def parse_frame_name(file_name: str) -> tuple[str, int | None]:
    """
    Returns (sequence, frame number). If there is no trailing frame number,
    the whole cleaned-up stem is the sequence and frame is None.
    """
    stem = PurePath(file_name).name
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem  # real extension
    stem = _ROBOFLOW_SUFFIX.sub("", stem)
    stem = _OLD_EXT.sub("", stem)
    m = _FRAME.match(stem)
    if m is None:
        return stem, None
    return m["seq"], int(m["frame"])


def sequence_of(file_name: str) -> str:
    return parse_frame_name(file_name)[0]
