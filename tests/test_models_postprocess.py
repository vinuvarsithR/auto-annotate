"""OWLv2 + SAM 2 steps 1-2 (merge_contained, hook_filter) against docs/fixtures/owl_sam/cases.json. CPU only."""

import json
from pathlib import Path

import pytest

from autolabel.models.base import Prediction
from autolabel.models.postprocess import CONTAIN_FRAC, hook_filter, merge_contained

CASES = json.loads((Path(__file__).resolve().parents[1] / "docs" / "fixtures" / "owl_sam" / "cases.json")
                   .read_text(encoding="utf-8"))


def preds(items):
    return [Prediction(image_id="img1", category=d["category"], bbox=tuple(d["bbox"]), score=d["score"],
                       source="owlv2_image") for d in items]


def assert_same(got, expected):
    assert [p.category for p in got] == [d["category"] for d in expected]
    for p, d in zip(got, expected):
        assert p.bbox == pytest.approx(tuple(d["bbox"]), abs=1e-6)
        assert p.score == pytest.approx(d["score"], abs=1e-6)
        assert p.image_id == "img1" and p.source == "owlv2_image"


@pytest.mark.parametrize("case", CASES["merge"], ids=[c["name"] for c in CASES["merge"]])
def test_merge_fixture(case):
    assert_same(merge_contained(preds(case["input"])), case["output"])


@pytest.mark.parametrize("case", CASES["hook_filter"], ids=[c["name"] for c in CASES["hook_filter"]])
def test_hook_filter_fixture(case):
    hook = tuple(case["hook"]) if case["hook"] is not None else None
    assert_same(hook_filter(preds(case["input"]), hook, case["load_classes"], case["radius"]), case["output"])


def test_contain_frac_is_contract_value():
    assert CONTAIN_FRAC == 0.8


def test_merge_keeps_other_fields_of_the_larger_box():
    big = Prediction("img1", "rb", (0, 0, 100, 100), 0.2, source="big", polygon=[[0, 0], [1, 0], [1, 1]])
    small = Prediction("img1", "rb", (10, 10, 20, 20), 0.9, source="small")
    (out,) = merge_contained([small, big])
    assert out.source == "big" and out.polygon == [[0, 0], [1, 0], [1, 1]] and out.score == 0.9


def test_merge_and_filter_do_not_change_the_input_list():
    items = preds(CASES["merge"][0]["input"])
    before = list(items)
    merge_contained(items)
    hook_filter(items, (616, 347), ["rb"], 130)
    assert items == before


def test_empty_inputs():
    assert merge_contained([]) == []
    assert hook_filter([], (1, 1), ["rb"], 130) == []
