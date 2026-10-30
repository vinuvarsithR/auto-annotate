"""Tests for the GPU-free prompt helpers in autolabel/gdino.py."""

from autolabel.gdino import build_prompt, token_owner


def test_build_prompt_lowercases_and_adds_dots():
    text, spans = build_prompt(["A Worker", "a long beam."])
    assert text == "a worker. a long beam."
    assert [text[a:b] for a, b in spans] == ["a worker", "a long beam"]


def test_token_owner_maps_tokens_to_phrases():
    text, spans = build_prompt(["a worker", "a beam"])  # "a worker. a beam."
    offsets = [(0, 0), (0, 1), (2, 8), (8, 9), (10, 11), (12, 16), (16, 17), (0, 0)]  # [CLS] a worker . a beam . [SEP]
    assert token_owner(offsets, spans) == [-1, 0, 0, -1, 1, 1, -1, -1]
