"""
dedup.py - spot images that look almost the same.

Plain-English summary
---------------------
* A "difference hash" (dHash) shrinks the image to a tiny greyscale grid and
  records, for each cell, whether it is brighter than its right-hand
  neighbour. That gives one bit per cell.
* Two images that look alike get hashes that differ in only a few bits.
  The number of differing bits is the Hamming distance.
* Our camera never moves, so even different scenes share most of the
  background. A bigger grid (16x16 = 256 bits) notices smaller changes
  such as a load or a worker moving.

No GPU, only Pillow + numpy.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

HASH_SIZE = 16  # 16 x 16 = 256 bits


def dhash(image: Image.Image, size: int = HASH_SIZE) -> int:
    """Difference hash of a PIL image, as a Python int with size*size bits."""
    grey = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = np.asarray(grey, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hamming(a: int, b: int) -> int:
    """Number of bits that differ between two hashes."""
    return (a ^ b).bit_count()
