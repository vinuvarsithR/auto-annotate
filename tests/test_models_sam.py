"""OWLv2 + SAM 2 steps 3-4 (clean_mask, largest_polygon, add_polygons) with numpy masks and FakeSegmenter.
No SAM weights needed. CPU only."""

import numpy as np
import pytest
from PIL import Image

from autolabel.models import sam
from autolabel.models.base import Prediction
from autolabel.models.sam import FakeSegmenter, add_polygons, clean_mask, get_segmenter, largest_polygon

H, W = 100, 200
BOX = (50, 20, 60, 40)  # x, y, w, h -> pixels x 50..110, y 20..60


def rect(x0, y0, x1, y1):
    """bool H x W mask, inclusive pixel ranges."""
    m = np.zeros((H, W), bool)
    m[y0:y1 + 1, x0:x1 + 1] = True
    return m


def pred(bbox=BOX, cls="rb", score=0.7):
    return Prediction("img1", cls, tuple(bbox), score, source="owlv2_image")


class OneMask:
    """Segmenter that returns the given (masks, scores) for every box, and records the boxes."""

    def __init__(self, masks, scores):
        self.masks, self.scores, self.calls = np.asarray(masks), np.asarray(scores, float), []

    def segment(self, image, boxes):
        self.calls.append(list(boxes))
        return [(self.masks, self.scores) for _ in boxes]


# ---------------- step 3: pick ----------------

def test_pick_highest_score_among_masks_inside_the_box():
    spills = rect(0, 0, 199, 99)          # whole image: far outside the box, best score
    inside_low = rect(60, 25, 100, 55)
    inside_high = rect(55, 22, 105, 58)
    out = clean_mask([spills, inside_low, inside_high], [0.99, 0.5, 0.8], BOX, open_px=0)
    assert np.array_equal(out, inside_high)


def test_inside_rule_uses_box_grown_by_5px():
    # 4 px outside the box on the left: still "inside" (grown box), so it wins on score; then clipped
    near = rect(46, 25, 100, 55)
    other = rect(60, 25, 100, 55)
    out = clean_mask([near, other], [0.9, 0.1], BOX, open_px=0)
    assert np.array_equal(out, rect(50, 25, 100, 55))


def test_no_mask_inside_falls_back_to_highest_inside_fraction():
    a = rect(0, 0, 199, 99)               # small share inside
    b = rect(30, 10, 120, 70)             # larger share inside, still < 90%
    c = np.zeros((H, W), bool)            # empty
    out = clean_mask([a, b, c], [0.9, 0.2, 0.99], BOX, open_px=0)
    assert np.array_equal(out, b & rect(50, 20, 110, 60))


def test_empty_masks_never_qualify():
    empty = np.zeros((H, W), bool)
    ok = rect(60, 25, 100, 55)
    out = clean_mask([empty, ok], [0.99, 0.1], BOX, open_px=0)
    assert np.array_equal(out, ok)


# ---------------- step 3: opening and clip ----------------

def test_opening_cuts_a_thin_strand():
    body = rect(60, 30, 90, 50)
    strand = rect(91, 40, 109, 41)        # 2 px thick cable
    out = clean_mask([body | strand], [0.9], BOX)
    assert not out[40:42, 95:110].any()   # strand gone
    assert out[35:46, 65:86].all()        # body interior kept


def test_opening_that_empties_keeps_the_mask_before_opening():
    tiny = rect(70, 30, 72, 32)           # 3 x 3: a 5 px opening removes it all
    out = clean_mask([tiny], [0.9], BOX)
    assert np.array_equal(out, tiny)


def test_clip_clears_pixels_outside_the_box():
    m = rect(47, 18, 113, 63)             # up to 3 px outside on every side
    out = clean_mask([m], [0.9], BOX, open_px=0)
    assert np.array_equal(out, rect(50, 20, 110, 60))


def test_clip_uses_floor_and_ceil_inclusive():
    m = rect(40, 10, 120, 70)
    out = clean_mask([m], [0.9], (50.4, 20.6, 59.2, 39.0), open_px=0)  # x 50.4..109.6, y 20.6..59.6
    assert np.array_equal(out, rect(50, 20, 110, 60))


# ---------------- step 4: polygon ----------------

def test_largest_polygon_of_a_rectangle():
    poly = largest_polygon(rect(50, 20, 109, 59), W, H)
    xs, ys = [p[0] * W for p in poly], [p[1] * H for p in poly]
    assert (min(xs), min(ys), max(xs), max(ys)) == pytest.approx((50, 20, 109, 59))
    assert all(v == round(v, 6) for p in poly for v in p)


def test_two_parts_give_the_largest_part_and_its_bbox():
    big, small = rect(60, 25, 100, 55), rect(103, 25, 108, 30)
    seg = OneMask([big | small], [0.9])
    (out,) = add_polygons(Image.new("RGB", (W, H)), [pred()], seg)
    assert out.polygon is not None and out.mask is None
    assert out.bbox == pytest.approx((60, 25, 40, 30))   # the big part only
    assert out.score == 0.7 and out.category == "rb" and out.source == "owlv2_image"


def test_empty_mask_keeps_the_box_without_polygon():
    seg = OneMask([np.zeros((H, W), bool)], [0.9])
    (out,) = add_polygons(Image.new("RGB", (W, H)), [pred()], seg)
    assert out.polygon is None and out.mask is None
    assert out.bbox == BOX and out.score == 0.7 and out.source == "owlv2_image"


@pytest.mark.parametrize("mask", [np.zeros((H, W), bool), rect(60, 30, 60, 50), rect(60, 30, 61, 30)],
                         ids=["empty", "1 px wide line", "2 px dot"])
def test_largest_polygon_none_when_not_usable(mask):
    assert largest_polygon(mask, W, H) is None


# ---------------- add_polygons / FakeSegmenter ----------------

def test_fake_segmenter_gives_an_ellipse_inside_each_box():
    image = Image.new("RGB", (W, H))
    (masks, scores), = FakeSegmenter().segment(image, [BOX])
    assert masks.shape == (1, H, W) and masks.dtype == bool and scores.tolist() == pytest.approx([0.9])
    ys, xs = np.nonzero(masks[0])
    assert (xs.min(), xs.max(), ys.min(), ys.max()) == (50, 110, 20, 60)
    assert not masks[0][20, 50] and masks[0][40, 80]       # corner out, center in


def test_add_polygons_with_fake_segmenter():
    image = Image.new("RGB", (W, H))
    out = add_polygons(image, [pred(), pred((10, 70, 30, 20), "person", 0.4)], FakeSegmenter())
    assert [p.category for p in out] == ["rb", "person"]
    for p, b in zip(out, [BOX, (10, 70, 30, 20)]):
        assert len(p.polygon) >= 3
        assert p.bbox == pytest.approx(b, abs=1.5)            # ellipse polygon box ~ the input box


def test_add_polygons_one_segment_call_per_image_and_none_for_no_boxes():
    seg = OneMask([rect(60, 25, 100, 55)], [0.9])
    image = Image.new("RGB", (W, H))
    assert add_polygons(image, [], seg) == [] and seg.calls == []
    add_polygons(image, [pred(), pred()], seg)
    assert seg.calls == [[BOX, BOX]]


# ---------------- weights ----------------

def test_missing_weights_raise_file_not_found(monkeypatch, tmp_path):
    monkeypatch.setenv(sam.ENV_WEIGHTS, str(tmp_path / "nope.pt"))
    with pytest.raises(FileNotFoundError, match="SAM weights not found"):
        get_segmenter()


def test_default_weights_path_is_models_folder(monkeypatch):
    monkeypatch.delenv(sam.ENV_WEIGHTS, raising=False)
    assert sam.DEFAULT_WEIGHTS.parent.name == "models" and sam.DEFAULT_WEIGHTS.name == "sam2.1_t.pt"
