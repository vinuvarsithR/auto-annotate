"""Tests for autolabel/thresholds.py, using hand-made sweep rows with known answers."""

from autolabel.evaluate import ALL, Box, sweep
from autolabel.thresholds import select_thresholds


def row(cls, t, recall, precision=0.5, n_gold=10):
    return {"threshold": t, "class": cls, "recall": recall, "precision": precision, "n_gold": n_gold}


def test_picks_highest_threshold_meeting_target():
    rows = [row("a", 0.05, 1.0), row("a", 0.10, 0.9), row("a", 0.15, 0.8)]
    c = select_thresholds(rows, 0.9)["a"]
    assert c["threshold"] == 0.10
    assert c["recall"] == 0.9  # exactly at target counts as met
    assert c["met"] is True


def test_reports_precision_at_chosen_threshold():
    rows = [row("a", 0.05, 1.0, precision=0.1), row("a", 0.10, 0.95, precision=0.4)]
    assert select_thresholds(rows, 0.9)["a"]["precision"] == 0.4


def test_target_not_met_falls_back_to_best_recall():
    rows = [row("a", 0.05, 0.6), row("a", 0.10, 0.4), row("a", 0.15, 0.2)]
    c = select_thresholds(rows, 0.9)["a"]
    assert c["threshold"] == 0.05
    assert c["recall"] == 0.6
    assert c["met"] is False


def test_fallback_tie_prefers_higher_threshold():
    # Same recall everywhere (e.g. 0.0): higher threshold means less junk.
    rows = [row("a", 0.05, 0.0), row("a", 0.10, 0.0), row("a", 0.15, 0.0)]
    c = select_thresholds(rows, 0.9)["a"]
    assert c["threshold"] == 0.15
    assert c["met"] is False


def test_non_monotonic_recall_still_picks_highest_meeting():
    rows = [row("a", 0.05, 0.95), row("a", 0.10, 0.85), row("a", 0.15, 0.92)]
    assert select_thresholds(rows, 0.9)["a"]["threshold"] == 0.15


def test_classes_are_chosen_independently():
    rows = [
        row("a", 0.05, 1.0), row("a", 0.10, 1.0),
        row("b", 0.05, 0.9), row("b", 0.10, 0.5),
    ]
    c = select_thresholds(rows, 0.9)
    assert c["a"]["threshold"] == 0.10
    assert c["b"]["threshold"] == 0.05


def test_class_without_gold_boxes_has_no_threshold():
    rows = [row("a", 0.05, None, n_gold=0), row("a", 0.10, None, n_gold=0)]
    c = select_thresholds(rows, 0.9)["a"]
    assert c["threshold"] is None
    assert c["met"] is False


def test_overall_row_is_skipped():
    rows = [row(ALL, 0.05, 1.0), row("a", 0.05, 1.0)]
    assert set(select_thresholds(rows, 0.9)) == {"a"}


def test_works_on_real_sweep_output():
    golds = [Box("im", "a", (0, 0, 10, 10)), Box("im", "a", (50, 50, 10, 10))]
    preds = [Box("im", "a", (0, 0, 10, 10), 0.30), Box("im", "a", (50, 50, 10, 10), 0.12)]
    rows = sweep(preds, golds, [0.05, 0.10, 0.15, 0.20, 0.30])
    c = select_thresholds(rows, 1.0)["a"]
    assert c["threshold"] == 0.10  # 0.15 would lose the 0.12 box
    assert c["recall"] == 1.0
    c = select_thresholds(rows, 0.5)["a"]
    assert c["threshold"] == 0.30


# ---------- apply_thresholds ----------

from autolabel.thresholds import apply_thresholds  # noqa: E402


def test_apply_thresholds_per_class():
    preds = [Box("im", "a", (0, 0, 1, 1), 0.3), Box("im", "a", (0, 0, 1, 1), 0.1),
             Box("im", "b", (0, 0, 1, 1), 0.3)]
    chosen = {"a": {"threshold": 0.2}, "b": {"threshold": 0.5}}
    assert [p.score for p in apply_thresholds(preds, chosen)] == [0.3]


def test_apply_thresholds_keeps_score_equal_to_threshold():
    assert len(apply_thresholds([Box("im", "a", (0, 0, 1, 1), 0.2)], {"a": {"threshold": 0.2}})) == 1


def test_apply_thresholds_drops_unknown_or_unset_class():
    preds = [Box("im", "a", (0, 0, 1, 1), 0.9), Box("im", "z", (0, 0, 1, 1), 0.9)]
    assert apply_thresholds(preds, {"a": {"threshold": None}}) == []
