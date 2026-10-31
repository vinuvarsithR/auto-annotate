"""Tests for pipeline.run() with the "fake" detector. CPU only."""

import pytest
from PIL import Image

from autolabel import pipeline
from autolabel.models.base import ClassSpec

CLASSES = [ClassSpec("person", "p"), ClassSpec("biax", "b")]


@pytest.fixture
def images(tmp_path):
    out = []
    for i, (w, h, mode) in enumerate([(64, 32, "RGB"), (50, 80, "L"), (100, 100, "RGBA")]):
        path = tmp_path / f"im{i}.png"
        Image.new(mode, (w, h)).save(path)
        out.append((f"id{i}", str(path)))
    return out


def test_one_result_per_image_in_order(images):
    results = list(pipeline.run(images, CLASSES, detector="fake"))
    assert [r[0] for r in results] == ["id0", "id1", "id2"]
    for image_id, preds in results:
        assert len(preds) == 2
        assert all(p.image_id == image_id for p in preds)


def test_boxes_use_each_image_size(images):
    results = dict(pipeline.run(images, CLASSES, detector="fake"))
    x, y, w, h = results["id1"][0].bbox  # image 1 is 50 x 80
    assert (x, y, w, h) == pytest.approx((5, 8, 15, 24))


def test_progress_callback(images):
    calls = []
    list(pipeline.run(images, CLASSES, detector="fake", on_progress=lambda d, t: calls.append((d, t))))
    assert calls == [(1, 3), (2, 3), (3, 3)]


def test_is_lazy_generator(images):
    calls = []
    gen = pipeline.run(images, CLASSES, detector="fake", on_progress=lambda d, t: calls.append(d))
    next(gen)
    assert calls == [1]


def test_empty_input():
    calls = []
    assert list(pipeline.run([], CLASSES, detector="fake", on_progress=lambda d, t: calls.append(d))) == []
    assert calls == []


def test_accepts_generator_input(images):
    results = list(pipeline.run((x for x in images), CLASSES, detector="fake"))
    assert len(results) == 3


def test_default_pipeline_not_available_yet(images):
    with pytest.raises(ValueError, match="not available yet"):
        list(pipeline.run(images, CLASSES))
