"""Tests for autolabel/dedup.py using small generated images."""

import numpy as np
from PIL import Image

from autolabel.dedup import dhash, hamming


def gradient(width=64, height=48, flip=False):
    row = np.linspace(0, 255, width, dtype=np.uint8)
    if flip:
        row = row[::-1]
    return Image.fromarray(np.tile(row, (height, 1)))


def test_same_image_same_hash():
    assert hamming(dhash(gradient()), dhash(gradient())) == 0


def test_resized_copy_is_close():
    a = gradient(64, 48)
    b = a.resize((128, 96))
    assert hamming(dhash(a), dhash(b)) <= 8


def test_different_image_is_far():
    assert hamming(dhash(gradient()), dhash(gradient(flip=True))) > 128


def test_hash_has_256_bits():
    assert dhash(gradient(flip=True)).bit_length() <= 256


def test_hamming_counts_bits():
    assert hamming(0b1011, 0b0001) == 2
