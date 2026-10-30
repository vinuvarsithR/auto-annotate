"""Tests for autolabel/gold.py using a tiny gold set written to a temp folder."""

import json

import pytest

from autolabel.gold import load_classes, load_gold_set

CLASSES_YAML = """
classes:
  - name: person
    description: "a person"
    aliases: [Person, a_person]
  - name: rb
    description: "a thing"
"""


def make_gold(tmp_path, classes_yaml=CLASSES_YAML):
    (tmp_path / "classes.yaml").write_text(classes_yaml)
    coco = {
        "images": [{"id": 0, "file_name": "a.jpg"}, {"id": 1, "file_name": "b.jpg"}],
        "categories": [
            {"id": 0, "name": "eot-crane"},  # Roboflow's project-name category
            {"id": 1, "name": "Person"},
            {"id": 2, "name": "a_person"},
            {"id": 3, "name": "rb"},
            {"id": 4, "name": "2"},  # junk
        ],
        "annotations": [
            {"id": 1, "image_id": 0, "category_id": 1, "bbox": [0, 0, 10, 10]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [5, 5, 10, 10]},
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [20, 20, 5, 5]},
            {"id": 4, "image_id": 1, "category_id": 4, "bbox": [1, 1, 1, 1]},
        ],
    }
    (tmp_path / "annotations.json").write_text(json.dumps(coco))
    return tmp_path


def test_aliases_map_to_canonical_name(tmp_path):
    g = load_gold_set(make_gold(tmp_path))
    assert sorted(b.category for b in g.boxes) == ["person", "person", "rb"]


def test_unknown_categories_are_skipped_and_counted(tmp_path):
    g = load_gold_set(make_gold(tmp_path))
    assert g.skipped_categories == {"2": 1}


def test_boxes_are_keyed_by_file_name(tmp_path):
    g = load_gold_set(make_gold(tmp_path))
    assert {b.image_id for b in g.boxes} == {"a.jpg", "b.jpg"}
    assert g.image_files == ["a.jpg", "b.jpg"]


def test_missing_description_is_an_error(tmp_path):
    (tmp_path / "classes.yaml").write_text('classes:\n  - name: rb\n    description: "TODO: describe"\n')
    with pytest.raises(ValueError, match="needs a description"):
        load_classes(tmp_path / "classes.yaml")


def test_alias_used_by_two_classes_is_an_error(tmp_path):
    bad = """
classes:
  - name: person
    description: "a person"
    aliases: [human]
  - name: worker
    description: "a worker"
    aliases: [human]
"""
    make_gold(tmp_path, bad)
    with pytest.raises(ValueError, match="listed under both"):
        load_gold_set(tmp_path)


def test_classes_override_replaces_gold_set_classes(tmp_path):
    (tmp_path / "gold").mkdir()
    gold_dir = make_gold(tmp_path / "gold")
    other = tmp_path / "other.yaml"
    other.write_text(CLASSES_YAML.replace('description: "a person"', 'description: "a hard hat"'))
    gold = load_gold_set(gold_dir, other)
    assert [c.description for c in gold.classes] == ["a hard hat", "a thing"]
    assert len(gold.boxes) == 3  # same gold boxes, same alias mapping


def test_examples_are_parsed_relative_to_yaml(tmp_path):
    (tmp_path / "ex").mkdir()
    yaml_text = CLASSES_YAML + """    examples:
      - {image: ex/a.jpg, bbox: [1, 2, 30, 40]}
"""
    (tmp_path / "c.yaml").write_text(yaml_text)
    rb = load_classes(tmp_path / "c.yaml")[1]
    assert rb.examples[0].image == (tmp_path / "ex" / "a.jpg").resolve()
    assert rb.examples[0].bbox == (1.0, 2.0, 30.0, 40.0)
    assert load_classes(tmp_path / "c.yaml")[0].examples == []


@pytest.mark.parametrize("example", ["{image: a.jpg}", "{image: a.jpg, bbox: [1, 2, 3]}", "{image: a.jpg, bbox: [1, 2, 0, 5]}"])
def test_bad_example_is_an_error(tmp_path, example):
    (tmp_path / "c.yaml").write_text(CLASSES_YAML + f"    examples:\n      - {example}\n")
    with pytest.raises(ValueError):
        load_classes(tmp_path / "c.yaml")
