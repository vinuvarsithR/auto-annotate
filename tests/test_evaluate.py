"""Tests for autolabel/evaluate.py, using small hand-made cases with known answers."""

import pytest

from autolabel.evaluate import ALL, Box, evaluate, iou, sweep


def G(img, cls, x, y, w, h):
    return Box(img, cls, (x, y, w, h), 1.0)


def P(img, cls, x, y, w, h, s):
    return Box(img, cls, (x, y, w, h), s)


# ---------- IoU ----------

def test_iou_identical_boxes_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_no_overlap_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 10, 10)) == 0.0


def test_iou_touching_edges_is_zero():
    assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0


def test_iou_half_overlap():
    # overlap 5x10 = 50, union 100 + 100 - 50 = 150
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)


def test_iou_box_inside_box():
    # small 5x5 = 25 inside big 10x10 = 100 -> 25 / 100
    assert iou((0, 0, 10, 10), (2, 2, 5, 5)) == pytest.approx(0.25)


def test_iou_zero_area_box():
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


# ---------- matching ----------

def test_perfect_prediction():
    r = evaluate([P("a", "person", 0, 0, 10, 10, 0.9)], [G("a", "person", 0, 0, 10, 10)])
    assert r["person"]["tp"] == 1
    assert r["person"]["precision"] == 1.0
    assert r["person"]["recall"] == 1.0


def test_missed_box_is_false_negative():
    r = evaluate([], [G("a", "person", 0, 0, 10, 10)])
    assert r["person"]["fn"] == 1
    assert r["person"]["recall"] == 0.0
    assert r["person"]["precision"] is None  # no predictions -> undefined


def test_extra_box_is_false_positive():
    r = evaluate([P("a", "person", 50, 50, 10, 10, 0.9)], [])
    assert r["person"]["fp"] == 1
    assert r["person"]["precision"] == 0.0
    assert r["person"]["recall"] is None  # nothing to find -> undefined


def test_duplicate_prediction_counts_once():
    # Two predictions on the same gold box: one TP, one FP.
    preds = [P("a", "person", 0, 0, 10, 10, 0.9), P("a", "person", 1, 1, 10, 10, 0.8)]
    r = evaluate(preds, [G("a", "person", 0, 0, 10, 10)])
    assert (r["person"]["tp"], r["person"]["fp"]) == (1, 1)


def test_low_iou_does_not_match():
    # IoU 1/3 < 0.5
    r = evaluate([P("a", "person", 5, 0, 10, 10, 0.9)], [G("a", "person", 0, 0, 10, 10)])
    assert (r["person"]["tp"], r["person"]["fp"], r["person"]["fn"]) == (0, 1, 1)


def test_iou_threshold_is_inclusive():
    # IoU exactly 0.5: overlap 10x10=100... use boxes with IoU = 0.5 exactly
    # gold 0..20 x 0..10 (200), pred 0..10 x 0..10 (100): inter 100, union 200
    r = evaluate([P("a", "x", 0, 0, 10, 10, 0.9)], [G("a", "x", 0, 0, 20, 10)])
    assert r["x"]["tp"] == 1


def test_boxes_in_different_images_never_match():
    r = evaluate([P("a", "person", 0, 0, 10, 10, 0.9)], [G("b", "person", 0, 0, 10, 10)])
    assert (r["person"]["tp"], r["person"]["fp"], r["person"]["fn"]) == (0, 1, 1)


def test_higher_score_claims_gold_first():
    # The 0.9 box overlaps gold slightly less than the 0.5 box,
    # but it still claims first; the 0.5 box becomes the FP.
    gold = [G("a", "x", 0, 0, 10, 10)]
    preds = [P("a", "x", 1, 0, 10, 10, 0.9), P("a", "x", 0, 0, 10, 10, 0.5)]
    r = evaluate(preds, gold, score_threshold=0.0)
    assert (r["x"]["tp"], r["x"]["fp"]) == (1, 1)
    # Raising the threshold to 0.6 removes the 0.5 box: still 1 TP, no FP.
    r = evaluate(preds, gold, score_threshold=0.6)
    assert (r["x"]["tp"], r["x"]["fp"]) == (1, 0)


# ---------- wrong class / class-ignored recall ----------

def test_wrong_class_is_fp_and_fn_but_found_when_class_ignored():
    gold = [G("a", "open_root", 0, 0, 10, 10)]
    preds = [P("a", "closed_root", 0, 0, 10, 10, 0.9)]
    r = evaluate(preds, gold)
    assert r["open_root"]["fn"] == 1
    assert r["closed_root"]["fp"] == 1
    assert r[ALL]["recall"] == 0.0
    assert r[ALL]["recall_class_ignored"] == 1.0


# ---------- overall row ----------

def test_overall_row_sums_classes():
    gold = [G("a", "person", 0, 0, 10, 10), G("a", "rb", 50, 50, 10, 10)]
    preds = [P("a", "person", 0, 0, 10, 10, 0.9), P("a", "rb", 200, 200, 10, 10, 0.9)]
    r = evaluate(preds, gold)
    assert (r[ALL]["tp"], r[ALL]["fp"], r[ALL]["fn"]) == (1, 1, 1)
    assert r[ALL]["precision"] == 0.5
    assert r[ALL]["recall"] == 0.5


def test_score_threshold_filters_predictions():
    preds = [P("a", "person", 0, 0, 10, 10, 0.1)]
    gold = [G("a", "person", 0, 0, 10, 10)]
    assert evaluate(preds, gold, score_threshold=0.05)["person"]["tp"] == 1
    assert evaluate(preds, gold, score_threshold=0.2)["person"]["tp"] == 0


# ---------- sweep ----------

def test_sweep_recall_never_increases_with_threshold():
    gold = [G("a", "x", i * 20, 0, 10, 10) for i in range(5)]
    preds = [P("a", "x", i * 20, 0, 10, 10, s) for i, s in enumerate([0.1, 0.2, 0.3, 0.4, 0.5])]
    rows = [r for r in sweep(preds, gold, [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]) if r["class"] == ALL]
    recalls = [r["recall"] for r in rows]
    assert recalls == sorted(recalls, reverse=True)
    assert recalls[0] == 1.0 and recalls[-1] == 0.0


def test_sweep_has_one_row_per_class_per_threshold():
    gold = [G("a", "x", 0, 0, 10, 10), G("a", "y", 50, 50, 10, 10)]
    rows = sweep([], gold, [0.1, 0.2])
    assert len(rows) == 2 * 3  # 2 thresholds x (x, y, ALL)


# ---------- AP / mAP@0.5 ----------

from autolabel.evaluate import average_precision, map50  # noqa: E402


def test_ap_perfect_is_one():
    g = [G("a", "x", 0, 0, 10, 10), G("b", "x", 0, 0, 10, 10)]
    p = [P("a", "x", 0, 0, 10, 10, 0.9), P("b", "x", 0, 0, 10, 10, 0.8)]
    assert average_precision(p, g) == pytest.approx(1.0)


def test_ap_no_predictions_is_zero():
    assert average_precision([], [G("a", "x", 0, 0, 10, 10)]) == 0.0


def test_ap_no_gold_is_none():
    assert average_precision([P("a", "x", 0, 0, 10, 10, 0.9)], []) is None


def test_ap_junk_ranked_first_halves_precision():
    g = [G("a", "x", 0, 0, 10, 10)]
    p = [P("a", "x", 50, 50, 10, 10, 0.9), P("a", "x", 0, 0, 10, 10, 0.5)]
    assert average_precision(p, g) == pytest.approx(0.5)


def test_ap_hit_miss_hit():
    # recall/precision after each: (0.5, 1), (0.5, 0.5), (1, 0.667) -> 0.5*1 + 0.5*0.667
    g = [G("a", "x", 0, 0, 10, 10), G("a", "x", 100, 100, 10, 10)]
    p = [P("a", "x", 0, 0, 10, 10, 0.9), P("a", "x", 50, 50, 10, 10, 0.8), P("a", "x", 100, 100, 10, 10, 0.7)]
    assert average_precision(p, g) == pytest.approx(0.5 + 0.5 * 2 / 3)


def test_ap_duplicate_counts_as_junk():
    g = [G("a", "x", 0, 0, 10, 10)]
    p = [P("a", "x", 0, 0, 10, 10, 0.9), P("a", "x", 0, 0, 10, 10, 0.8)]
    assert average_precision(p, g) == pytest.approx(1.0)  # junk after full recall doesn't lower AP


def test_ap_does_not_depend_on_score_scale():
    g = [G("a", "x", 0, 0, 10, 10), G("a", "x", 100, 100, 10, 10)]
    p1 = [P("a", "x", 0, 0, 10, 10, 0.9), P("a", "x", 50, 50, 10, 10, 0.8), P("a", "x", 100, 100, 10, 10, 0.7)]
    p2 = [Box(b.image_id, b.category, b.bbox, b.score / 100) for b in p1]
    assert average_precision(p1, g) == pytest.approx(average_precision(p2, g))


def test_map50_averages_classes_with_gold():
    g = [G("a", "x", 0, 0, 10, 10), G("a", "y", 0, 0, 10, 10)]
    p = [P("a", "x", 0, 0, 10, 10, 0.9), P("a", "z", 0, 0, 10, 10, 0.9)]
    m = map50(p, g)
    assert m["x"] == pytest.approx(1.0) and m["y"] == 0.0
    assert "z" not in m
    assert m[ALL] == pytest.approx(0.5)
