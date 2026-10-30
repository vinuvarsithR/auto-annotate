"""Tests for autolabel/sequences.py: file names as they appear raw and in Roboflow exports."""

import pytest

from autolabel.sequences import parse_frame_name, sequence_of


@pytest.mark.parametrize(
    "name, seq, frame",
    [
        ("rb_load__cam0_2026-09-29_18-24-11_00568.jpg", "rb_load__cam0_2026-09-29_18-24-11", 568),
        (
            "rb_load__cam0_2026-09-29_18-24-11_00568_jpg.rf.c8af900d0d06c05cb817324b22898967.jpg",
            "rb_load__cam0_2026-09-29_18-24-11",
            568,
        ),
        ("biax__cam0_2026-09-29_18-48-25_00101_jpg.rf.ef54.jpg", "biax__cam0_2026-09-29_18-48-25", 101),
        ("sb_evener__cam0_2026-09-29_18-57-19_00001.PNG", "sb_evener__cam0_2026-09-29_18-57-19", 1),
        ("rb_1__cam0_2026-09-29_16-34-20_00299_jpeg.rf.AB12.jpg", "rb_1__cam0_2026-09-29_16-34-20", 299),
        ("lm_cpe_2026-09-15_15-37-00__cam0_t0036.jpg", "lm_cpe_2026-09-15_15-37-00__cam0", 36),
        ("rootMainLamination__cam0_00748.jpg", "rootMainLamination__cam0", 748),
        ("some/folder/rb_1__cam0_2026-09-29_16-34-20_00299.jpg", "rb_1__cam0_2026-09-29_16-34-20", 299),
    ],
)
def test_parse_frame_name(name, seq, frame):
    assert parse_frame_name(name) == (seq, frame)


def test_raw_and_roboflow_names_give_same_sequence():
    raw = "rb_load__cam0_2026-09-29_18-24-11_00568.jpg"
    exported = "rb_load__cam0_2026-09-29_18-24-11_00570_jpg.rf.0123abcd.jpg"
    assert sequence_of(raw) == sequence_of(exported)


def test_different_start_times_are_different_sequences():
    a = "rb_1__cam0_2026-09-29_16-24-22_00422.jpg"
    b = "rb_1__cam0_2026-09-29_16-34-20_00422.jpg"
    assert sequence_of(a) != sequence_of(b)


def test_name_without_frame_number_is_its_own_sequence():
    assert parse_frame_name("snapshot.jpg") == ("snapshot", None)
    assert parse_frame_name("snapshot_jpg.rf.abc123.jpg") == ("snapshot", None)


def test_time_part_is_not_mistaken_for_frame():
    # '18-24-11' has dashes, so only the final '_00568' is the frame.
    seq, frame = parse_frame_name("x__cam0_2026-09-29_18-24-11_00568.jpg")
    assert seq.endswith("18-24-11") and frame == 568
