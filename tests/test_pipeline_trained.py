"""pipeline.run accepts a name registered with register_trained (CONTRACT #9, Train on Colab)."""

import numpy as np
import pytest
from PIL import Image

from autolabel import pipeline
from autolabel.models import ClassSpec, register_trained, unregister_trained


class _Boxes:
    def __init__(self, rows):
        a = np.array(rows, dtype=float).reshape(-1, 6)
        self.xyxy, self.conf, self.cls = a[:, :4], a[:, 4], a[:, 5]


class _Result:
    def __init__(self, rows):
        self.boxes, self.masks = _Boxes(rows), None


class _Model:
    task, names, ckpt, overrides = "detect", {0: "person"}, {}, {}

    def predict(self, image, **kw):
        return [_Result([(1, 2, 11, 22, 0.8, 0)])]


@pytest.fixture
def image_path(tmp_path):
    p = tmp_path / "a.jpg"
    Image.new("RGB", (40, 30)).save(p)
    return str(p)


def test_run_with_registered_trained_name(image_path, tmp_path):
    register_trained("proj_v1", str(tmp_path / "w.pt"), "detect", model=_Model())
    try:
        out = list(pipeline.run([("5", image_path)], [ClassSpec("person", "")], detector="proj_v1"))
    finally:
        unregister_trained("proj_v1")
    (image_id, preds), = out
    assert image_id == "5"
    assert [(p.category, p.bbox, p.source) for p in preds] == [("person", (1.0, 2.0, 10.0, 20.0), "yolo_trained")]


def test_run_after_unregister_is_unknown(image_path, tmp_path):
    register_trained("proj_v1", str(tmp_path / "w.pt"), "detect", model=_Model())
    unregister_trained("proj_v1")
    with pytest.raises(ValueError):
        list(pipeline.run([("5", image_path)], [ClassSpec("person", "")], detector="proj_v1"))
