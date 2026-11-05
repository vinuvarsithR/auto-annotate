"""Polygons on #12 (add) and #13 (patch): validation, bbox from polygon, the #13 table.

Images are 1280x720 (the helper's default), so 0.5 -> x 640 and 0.25 -> y 180.
"""


import pytest

from tests.test_api_fixture_keys import FIXTURES
from tests.test_api_helpers import make_client, setup_project

TRIANGLE = [[0.5, 0.25], [0.75, 0.25], [0.625, 0.5]]
TRIANGLE_BBOX = [640.0, 180.0, 320.0, 180.0]

# Raw JSON text, because NaN and huge numbers cannot go through the client's json=.
BAD_POLYGONS = {
    "empty": "[]",
    "two_points": "[[0.1, 0.1], [0.2, 0.2]]",
    "not_a_list": '"abc"',
    "dict": '{"points": []}',
    "number": "5",
    "point_not_a_list": "[0.1, 0.2, 0.3]",
    "point_is_dict": '[{"x": 0.1, "y": 0.1}, [0.2, 0.1], [0.2, 0.2]]',
    "point_one_number": "[[0.1], [0.2, 0.1], [0.2, 0.2]]",
    "point_three_numbers": "[[0.1, 0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]",
    "string_number": '[["0.1", 0.1], [0.2, 0.1], [0.2, 0.2]]',
    "bool": "[[true, 0.1], [0.2, 0.1], [0.2, 0.2]]",
    "null_number": "[[null, 0.1], [0.2, 0.1], [0.2, 0.2]]",
    "nan": "[[NaN, 0.1], [0.2, 0.1], [0.2, 0.2]]",
    "infinity": "[[Infinity, 0.1], [0.2, 0.1], [0.2, 0.2]]",
    "overflows_to_infinity": "[[1e999, 0.1], [0.2, 0.1], [0.2, 0.2]]",
    "huge_integer": f"[[1{'0' * 400}, 0.1], [0.2, 0.1], [0.2, 0.2]]",
    "above_one": "[[1.01, 0.1], [0.2, 0.1], [0.2, 0.2]]",
    "below_zero": "[[0.1, -0.01], [0.2, 0.1], [0.2, 0.2]]",
    "pixels_not_normalized": "[[100, 200], [300, 200], [200, 400]]",
    "points_in_a_line": "[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]",
    "same_point_three_times": "[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]",
    "crossing_shape_with_zero_area": "[[0.1, 0.1], [0.3, 0.3], [0.3, 0.1], [0.1, 0.3]]",
}


def _setup(c):
    s = setup_project(c, n_images=1)
    s["iid"] = s["image_ids"][0]
    s["cid"] = s["class_ids"]["rb"]
    return s


def _add(c, s, **body):
    return c.post(f"/api/images/{s['iid']}/annotations", json={"class_id": s["cid"], **body})


def _patch(c, aid, **body):
    return c.patch(f"/api/annotations/{aid}", json=body)


def _polygon_row(c, s):
    r = _add(c, s, polygon=TRIANGLE)
    assert r.status_code == 201, r.text
    return r.json()


def _box_row(c, s):
    r = _add(c, s, bbox=[100, 200, 40, 50])
    assert r.status_code == 201, r.text
    return r.json()


def _approx(poly):
    return [pytest.approx(p, abs=1e-9) for p in poly]


# ------------------------------------------------------------------ every 400 rule

@pytest.mark.parametrize("text", BAD_POLYGONS.values(), ids=BAD_POLYGONS.keys())
def test_bad_polygon_on_add_is_400_and_stores_nothing(tmp_path, text):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        r = c.post(f"/api/images/{s['iid']}/annotations",
                   content=f'{{"class_id": {s["cid"]}, "polygon": {text}}}',
                   headers={"content-type": "application/json"})
        assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_request")
        assert c.get(f"/api/images/{s['iid']}/annotations").json()["annotations"] == []


@pytest.mark.parametrize("text", BAD_POLYGONS.values(), ids=BAD_POLYGONS.keys())
def test_bad_polygon_on_patch_is_400_and_changes_nothing(tmp_path, text):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        before = _polygon_row(c, s)
        r = c.patch(f"/api/annotations/{before['id']}", content=f'{{"polygon": {text}}}',
                    headers={"content-type": "application/json"})
        assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_request")
        after = c.get(f"/api/images/{s['iid']}/annotations").json()["annotations"]
        assert after == [before]


def test_bad_polygon_is_400_even_with_a_good_bbox(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        assert _add(c, s, polygon=[[0.1, 0.1]], bbox=[1, 2, 3, 4]).status_code == 400


def test_add_needs_bbox_or_polygon(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        for body in ({}, {"polygon": None}, {"bbox": None}, {"bbox": None, "polygon": None}):
            r = _add(c, s, **body)
            assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_request"), body


# ------------------------------------------------------------------ #12

def test_add_computes_bbox_from_polygon(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        a = _polygon_row(c, s)
        assert a["bbox"] == pytest.approx(TRIANGLE_BBOX)
        assert a["polygon"] == TRIANGLE
        assert (a["source"], a["score"], a["status"], a["mask"]) == ("human", None, "accepted", None)
        listed = c.get(f"/api/images/{s['iid']}/annotations").json()["annotations"]
        assert listed == [a]


def test_add_ignores_a_bbox_sent_with_polygon(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        for bbox in ([1, 2, 3, 4], [1, 2, 0, 0], [1, 2], "junk"):  # even an invalid one
            r = _add(c, s, polygon=TRIANGLE, bbox=bbox)
            assert r.status_code == 201, r.text
            assert r.json()["bbox"] == pytest.approx(TRIANGLE_BBOX)


def test_add_accepts_integer_corners_of_the_image(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        a = _add(c, s, polygon=[[0, 0], [1, 0], [1, 1], [0, 1]]).json()
        assert a["bbox"] == [0.0, 0.0, 1280.0, 720.0]
        assert a["polygon"] == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def test_add_without_polygon_is_a_plain_box(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        for body in ({"bbox": [100, 200, 40, 50]},
                     {"bbox": [100, 200, 40, 50], "polygon": None}):
            a = _add(c, s, **body).json()
            assert a["bbox"] == [100.0, 200.0, 40.0, 50.0] and a["polygon"] is None


def test_add_polygon_on_empty_image_is_409(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        assert c.patch(f"/api/images/{s['iid']}", json={"review_status": "empty"}).status_code == 200
        r = _add(c, s, polygon=TRIANGLE)
        assert (r.status_code, r.json()["error"]["code"]) == (409, "conflict")


# ------------------------------------------------------------------ #13 polygon sent

def test_patch_polygon_replaces_it_and_recomputes_bbox(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        aid = _polygon_row(c, s)["id"]
        new = [[0.25, 0.5], [0.5, 0.5], [0.5, 1.0]]
        a = _patch(c, aid, polygon=new).json()
        assert a["polygon"] == new and a["status"] == "edited"
        assert a["bbox"] == pytest.approx([320.0, 360.0, 320.0, 360.0])


def test_patch_polygon_turns_a_box_into_a_polygon(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        a = _patch(c, _box_row(c, s)["id"], polygon=TRIANGLE).json()
        assert a["polygon"] == TRIANGLE and a["status"] == "edited"
        assert a["bbox"] == pytest.approx(TRIANGLE_BBOX)


def test_patch_ignores_a_bbox_sent_with_polygon(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        aid = _box_row(c, s)["id"]
        for bbox in ([1, 2, 3, 4], [1, 2, 0, 0], "junk"):
            a = _patch(c, aid, polygon=TRIANGLE, bbox=bbox)
            assert a.status_code == 200, a.text
            assert a.json()["bbox"] == pytest.approx(TRIANGLE_BBOX)


def test_patch_same_polygon_is_not_an_edit(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        a = _polygon_row(c, s)
        assert _patch(c, a["id"], polygon=TRIANGLE).json()["status"] == "accepted"


def test_explicit_status_wins_over_polygon_change(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        aid = _polygon_row(c, s)["id"]
        a = _patch(c, aid, polygon=[[0, 0], [0.5, 0], [0, 0.5]], status="needs_review").json()
        assert a["status"] == "needs_review"


def test_undo_puts_polygon_box_and_status_back(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        old = _polygon_row(c, s)
        _patch(c, old["id"], bbox=[0, 0, 100, 100])
        back = _patch(c, old["id"], polygon=old["polygon"], status=old["status"]).json()
        assert back["polygon"] == old["polygon"] and back["status"] == old["status"]
        assert back["bbox"] == pytest.approx(old["bbox"])


# ------------------------------------------------------------------ #13 bbox only

def test_bbox_only_patch_scales_and_moves_polygon(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        aid = _polygon_row(c, s)["id"]
        a = _patch(c, aid, bbox=[0, 0, 640, 90]).json()  # x 2x wider, y half as tall
        assert a["polygon"] == _approx([[0.0, 0.0], [0.5, 0.0], [0.25, 0.125]])
        assert a["bbox"] == pytest.approx([0.0, 0.0, 640.0, 90.0])
        assert a["status"] == "edited"


def test_bbox_only_patch_matches_the_fixture_numbers(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        old = FIXTURES["GET /api/images/11/annotations"]["annotations"]
        old = next(a for a in old if a["id"] == 106)
        aid = _add(c, s, polygon=old["polygon"]).json()["id"]
        want = FIXTURES["PATCH /api/annotations/106 (bbox only, polygon scaled)"]
        a = _patch(c, aid, bbox=want["bbox"]).json()
        assert a["polygon"] == [pytest.approx(p, abs=2e-6) for p in want["polygon"]]
        assert a["bbox"] == pytest.approx(want["bbox"], abs=1e-3)


def test_bbox_only_patch_clamps_points_and_recomputes_bbox(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        aid = _polygon_row(c, s)["id"]
        a = _patch(c, aid, bbox=[1000, -90, 640, 180]).json()  # past the right and top edges
        assert a["polygon"] == _approx([[0.78125, 0.0], [1.0, 0.0], [1.0, 0.125]])
        assert a["bbox"] == pytest.approx([1000.0, 0.0, 280.0, 90.0])  # the clamped shape's box


@pytest.mark.parametrize("bbox", [[2000, 100, 50, 50], [100, -500, 50, 50]],
                         ids=["right_of_image", "above_image"])
def test_bbox_only_patch_outside_the_image_is_400_and_changes_nothing(tmp_path, bbox):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        before = _polygon_row(c, s)
        r = _patch(c, before["id"], bbox=bbox)
        assert (r.status_code, r.json()["error"]["code"]) == (400, "bad_request")
        assert c.get(f"/api/images/{s['iid']}/annotations").json()["annotations"] == [before]


def test_bbox_only_patch_with_nan_on_polygon_row_is_400(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        aid = _polygon_row(c, s)["id"]
        r = c.patch(f"/api/annotations/{aid}", content='{"bbox": [NaN, 0, 10, 10]}',
                    headers={"content-type": "application/json"})
        assert r.status_code == 400


def test_same_bbox_on_polygon_row_changes_nothing(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        before = _polygon_row(c, s)
        after = _patch(c, before["id"], bbox=before["bbox"]).json()
        assert after["status"] == "accepted" and after["polygon"] == TRIANGLE


def test_bbox_only_patch_on_plain_box_stays_a_plain_box(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        a = _patch(c, _box_row(c, s)["id"], bbox=[1, 2, 3, 4]).json()
        assert a["bbox"] == [1.0, 2.0, 3.0, 4.0] and a["polygon"] is None
        assert a["status"] == "edited"


# ------------------------------------------------------------------ #13 polygon: null / absent

def test_polygon_null_without_bbox_removes_it_and_keeps_the_bbox(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        old = _polygon_row(c, s)
        a = _patch(c, old["id"], polygon=None).json()
        assert a["polygon"] is None and a["bbox"] == old["bbox"] and a["status"] == "edited"


def test_polygon_null_with_bbox_applies_the_bbox(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        aid = _polygon_row(c, s)["id"]
        a = _patch(c, aid, polygon=None, bbox=[1, 2, 3, 4]).json()
        assert a["polygon"] is None and a["bbox"] == [1.0, 2.0, 3.0, 4.0]
        assert a["status"] == "edited"


def test_polygon_null_with_bad_bbox_is_400(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        aid = _polygon_row(c, s)["id"]
        assert _patch(c, aid, polygon=None, bbox=[1, 2, 0, 4]).status_code == 400


def test_polygon_null_on_plain_box_is_not_an_edit(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        old = _box_row(c, s)
        a = _patch(c, old["id"], polygon=None).json()
        assert a["status"] == "accepted" and a["bbox"] == old["bbox"]


def test_polygon_absent_leaves_it_untouched(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        old = _polygon_row(c, s)
        a = _patch(c, old["id"], class_id=s["class_ids"]["sb"], reviewer="dhyan").json()
        assert a["polygon"] == TRIANGLE and a["bbox"] == old["bbox"]
        assert a["class_name"] == "sb"
        a = _patch(c, old["id"], status="rejected").json()
        assert a["polygon"] == TRIANGLE and a["status"] == "rejected"


# ------------------------------------------------------------------ fixture keys

def test_sample_post_body_gives_the_sample_response(tmp_path):
    with make_client(tmp_path)[0] as c:
        s = _setup(c)
        want = FIXTURES["POST /api/images/11/annotations (polygon)"]
        a = _add(c, s, polygon=want["polygon"], reviewer="dhyan").json()
        assert a["bbox"] == pytest.approx(want["bbox"]) and a["polygon"] == want["polygon"]
