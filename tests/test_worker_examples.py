"""#9 examples mode and the mode field. Nothing here depends on models' fake examples support:
pipeline.prepare_examples, pipeline.run and detector_modes are monkeypatched, and the app uses
the real default loaders, so the wiring to autolabel.pipeline is tested too."""

from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import autolabel.models
import autolabel.models.base
import autolabel.pipeline
from app.db import db
from app.main import create_app
from app.worker.worker import clip_box, evenly_spaced
from tests.test_api_fixture_keys import FIXTURES, same_keys
from tests.test_api_helpers import FakePrediction, make_client, setup_project

TRIANGLE = [[0.5, 0.25], [0.75, 0.25], [0.625, 0.5]]


@dataclass(frozen=True)
class StandInExample:  # same fields as the contract's Example, used until models merges it
    image_id: str
    path: str
    category: str
    bbox: tuple


class FakeExampleRun:
    def __init__(self, classes, examples, unusable, min_examples=None):
        if min_examples:  # like owlv2: a class with too few examples is not run
            unusable = set(unusable) | {s.name for s in classes
                                        if len(examples[s.name]) < min_examples}
            self.skip_reasons = {s.name: f"needs at least {min_examples} examples"
                                 for s in classes if len(examples[s.name]) < min_examples}
        self.classes = [s for s in classes if s.name not in unusable]
        self.used = {s.name: len(examples[s.name]) for s in self.classes}
        self.unusable = [s.name for s in classes if s.name in unusable]


class Models:
    """Stands in for autolabel.pipeline. Each image gets one box per class that runs."""

    def __init__(self):
        self.unusable = ()    # class names the detector rejects
        self.min_examples = None  # set: the fake run gives skip_reasons like owlv2
        self.prepare_error = None
        self.prepare_calls = []
        self.run_calls = []
        self.polygon = None
        self.extra = []       # extra FakePrediction args, per image
        self.on_run = None

    def prepare(self, detector, classes, examples, options=None):
        self.prepare_calls.append({"detector": detector, "classes": classes,
                                   "examples": examples, "options": options})
        if self.prepare_error:
            raise self.prepare_error
        return FakeExampleRun(classes, examples, self.unusable, self.min_examples)

    def run(self, images, classes, **kwargs):
        images = list(images)
        self.run_calls.append({"images": images, "classes": classes, "kwargs": kwargs})
        if self.on_run:
            self.on_run()
        for image_id, _path in images:
            preds = [FakePrediction(image_id, s.name, (600, 300, 40, 40), 0.3, "new_model",
                                    None, self.polygon) for s in classes]
            yield image_id, preds + [FakePrediction(image_id, *e) for e in self.extra]


@pytest.fixture
def models(monkeypatch):
    m = Models()
    monkeypatch.setattr(autolabel.pipeline, "prepare_examples", m.prepare, raising=False)
    monkeypatch.setattr(autolabel.pipeline, "run", m.run)
    monkeypatch.setattr(autolabel.models, "detector_modes",
                        lambda name: {"fake": ["text", "examples"]}.get(name, []), raising=False)
    if not hasattr(autolabel.models.base, "Example"):
        monkeypatch.setattr(autolabel.models.base, "Example", StandInExample, raising=False)
    return m


@pytest.fixture
def client(tmp_path, models):
    app = create_app(tmp_path / "data", start_worker=False, list_detectors=lambda: ["fake"],
                     static_dir=None)
    with TestClient(app) as c:
        yield c


def _insert(c, iid, cid, status="accepted", source="human", bbox=(10, 20, 30, 40), job_id=None):
    conn = db.connect(c.app.state.db_path)
    try:
        with conn:
            return conn.execute(
                "INSERT INTO annotations (image_id, class_id, x, y, w, h, source, status, job_id, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 't', 't')",
                (iid, cid, *bbox, source, status, job_id)).lastrowid
    finally:
        conn.close()


def _job(c, pid, **body):
    r = c.post(f"/api/projects/{pid}/jobs", json={"model": "fake", "mode": "examples", **body})
    assert r.status_code == 202, r.text
    return r.json()


def _run(c, pid, **body):
    job = _job(c, pid, **body)
    assert c.app.state.worker.run_once()
    return c.get(f"/api/jobs/{job['id']}").json()


def _rows(c, iid):
    return c.get(f"/api/images/{iid}/annotations").json()["annotations"]


def _sent(models, name):
    return models.prepare_calls[0]["examples"].get(name, [])


def _image_path(c, iid):
    conn = db.connect(c.app.state.db_path)
    try:
        return conn.execute("SELECT path FROM images WHERE id=?", (iid,)).fetchone()[0]
    finally:
        conn.close()


# ------------------------------------------------------------------ which boxes are examples

@pytest.mark.parametrize("source", ["human", "import", "owlv2"])
@pytest.mark.parametrize("status", ["accepted", "edited"])
def test_reviewed_box_of_any_source_is_an_example(client, models, status, source):
    s = setup_project(client, n_images=1)
    _insert(client, s["image_ids"][0], s["class_ids"]["rb"], status, source)
    _run(client, s["pid"])
    assert [e.bbox for e in _sent(models, "rb")] == [(10.0, 20.0, 30.0, 40.0)]


@pytest.mark.parametrize("source", ["human", "import", "owlv2"])
@pytest.mark.parametrize("status", ["rejected", "auto_accepted", "needs_review"])
def test_unreviewed_or_rejected_box_of_any_source_is_not(client, models, status, source):
    s = setup_project(client, n_images=1)
    _insert(client, s["image_ids"][0], s["class_ids"]["rb"], status, source)
    job = _run(client, s["pid"])
    assert models.prepare_calls == []
    assert {"class_name": "rb", "reason": "no examples"} in job["result"]["skipped_classes"]


def test_examples_come_from_all_images_whatever_image_ids_says(client, models):
    s = setup_project(client, n_images=2)
    first, second = s["image_ids"]
    _insert(client, second, s["class_ids"]["rb"])
    _run(client, s["pid"], image_ids=[first])
    example = _sent(models, "rb")[0]
    assert (example.image_id, example.path, example.category) == \
        (str(second), _image_path(client, second), "rb")
    assert [i for i, _ in models.run_calls[0]["images"]] == [str(first)]


def test_examples_from_another_project_are_not_used(client, models):
    s = setup_project(client, n_images=1)
    other = setup_project(client, n_images=1, name="other")
    _insert(client, other["image_ids"][0], other["class_ids"]["rb"])
    _insert(client, s["image_ids"][0], s["class_ids"]["rb"], bbox=(1, 2, 3, 4))
    _run(client, s["pid"])
    assert [e.bbox for e in _sent(models, "rb")] == [(1.0, 2.0, 3.0, 4.0)]


def test_polygon_and_mask_on_an_example_are_ignored(client, models):
    s = setup_project(client, n_images=1)
    iid = s["image_ids"][0]
    a = client.post(f"/api/images/{iid}/annotations",
                    json={"class_id": s["class_ids"]["rb"], "polygon": TRIANGLE,
                          "mask": {"size": [720, 1280], "counts": "x"}}).json()
    _run(client, s["pid"])
    assert _sent(models, "rb")[0].bbox == pytest.approx(tuple(a["bbox"]))


# ------------------------------------------------------------------ cap

def test_more_than_30_are_capped_evenly_in_id_order(client, models):
    s = setup_project(client, n_images=1)
    iid, cid = s["image_ids"][0], s["class_ids"]["rb"]
    for i in range(60):
        _insert(client, iid, cid, bbox=(i, 0, 10, 10))
    job = _run(client, s["pid"])
    assert [e.bbox[0] for e in _sent(models, "rb")] == [float(i) for i in range(0, 60, 2)]
    assert job["result"]["examples_by_class"] == {"rb": 30}


def test_exactly_30_are_all_kept(client, models):
    s = setup_project(client, n_images=1)
    for i in range(30):
        _insert(client, s["image_ids"][0], s["class_ids"]["rb"], bbox=(i, 0, 10, 10))
    _run(client, s["pid"])
    assert [e.bbox[0] for e in _sent(models, "rb")] == [float(i) for i in range(30)]


def test_cap_is_per_class(client, models):
    s = setup_project(client, n_images=1)
    for i in range(31):
        _insert(client, s["image_ids"][0], s["class_ids"]["rb"], bbox=(i, 0, 10, 10))
    _insert(client, s["image_ids"][0], s["class_ids"]["sb"])
    job = _run(client, s["pid"])
    assert job["result"]["examples_by_class"] == {"rb": 30, "sb": 1}


def test_cap_counts_only_examples_left_after_clean_up(client, models):
    s = setup_project(client, n_images=1)
    iid, cid = s["image_ids"][0], s["class_ids"]["rb"]
    for _ in range(40):
        _insert(client, iid, cid, bbox=(5000, 0, 10, 10))  # outside the image
    for i in range(30):
        _insert(client, iid, cid, bbox=(i, 0, 10, 10))
    _run(client, s["pid"])
    assert [e.bbox[0] for e in _sent(models, "rb")] == [float(i) for i in range(30)]


@pytest.mark.parametrize("n, cap", [(31, 30), (45, 30), (59, 30), (1000, 30), (7, 3)])
def test_evenly_spaced_keeps_order_and_even_gaps(n, cap):
    picked = evenly_spaced(list(range(n)), cap)
    assert len(picked) == cap and picked[0] == 0
    gaps = {b - a for a, b in zip(picked, picked[1:])}
    assert min(gaps) >= 1 and max(gaps) - min(gaps) <= 1


def test_evenly_spaced_leaves_short_lists_alone():
    assert evenly_spaced([1, 2], 30) == [1, 2]


# ------------------------------------------------------------------ clean-up

@pytest.mark.parametrize("box, clipped", [
    ((-10, -20, 30, 40), (0.0, 0.0, 20.0, 20.0)),
    ((1270, 710, 50, 50), (1270.0, 710.0, 10.0, 10.0)),
    ((-5, -5, 2000, 2000), (0.0, 0.0, 1280.0, 720.0)),
    ((1, 2, 3, 4), (1.0, 2.0, 3.0, 4.0)),
])
def test_example_bbox_is_clipped_to_the_image(client, models, box, clipped):
    s = setup_project(client, n_images=1)
    _insert(client, s["image_ids"][0], s["class_ids"]["rb"], bbox=box)
    _run(client, s["pid"])
    assert _sent(models, "rb")[0].bbox == clipped


@pytest.mark.parametrize("box", [(2000, 10, 5, 5), (10, 800, 5, 5), (-50, 10, 50, 5), (1280, 0, 5, 5)])
def test_example_with_no_area_left_is_dropped(client, models, box):
    s = setup_project(client, n_images=1)
    _insert(client, s["image_ids"][0], s["class_ids"]["rb"], bbox=box)
    job = _run(client, s["pid"])
    assert models.prepare_calls == []
    assert {"class_name": "rb", "reason": "no examples"} in job["result"]["skipped_classes"]


def test_clip_box_treats_nan_as_no_area():
    assert clip_box(float("nan"), 0, 10, 10, 100, 100) is None


def test_example_with_missing_file_is_dropped_others_stay(client, models):
    s = setup_project(client, n_images=2)
    gone, kept = s["image_ids"]
    cid = s["class_ids"]["rb"]
    _insert(client, gone, cid)
    _insert(client, kept, cid)
    Path(_image_path(client, gone)).unlink()
    job = _run(client, s["pid"], image_ids=[kept])
    assert [e.image_id for e in _sent(models, "rb")] == [str(kept)]
    assert job["status"] == "done" and job["result"]["examples_by_class"] == {"rb": 1}


def test_class_with_only_missing_files_is_skipped(client, models):
    s = setup_project(client, n_images=1)
    iid = s["image_ids"][0]
    _insert(client, iid, s["class_ids"]["rb"])
    Path(_image_path(client, iid)).unlink()
    job = _run(client, s["pid"])
    assert job["status"] == "done" and models.prepare_calls == []


# ------------------------------------------------------------------ mode in #9

@pytest.mark.parametrize("mode", ["nope", "", "Examples", 5, True, ["text"], {"a": 1}])
def test_unknown_mode_is_400_and_creates_no_job(client, models, mode):
    s = setup_project(client, n_images=1)
    r = client.post(f"/api/projects/{s['pid']}/jobs", json={"model": "fake", "mode": mode})
    assert (r.status_code, r.json()["error"]["code"]) == (400, "unknown_mode")
    assert client.app.state.worker.run_once() is False


def test_any_mode_on_a_detector_without_modes_is_unknown_mode(tmp_path):
    with make_client(tmp_path, detectors=("yolo",), modes={"yolo": []})[0] as c:
        s = setup_project(c, n_images=1)
        r = c.post(f"/api/projects/{s['pid']}/jobs", json={"model": "yolo", "mode": "text"})
        assert (r.status_code, r.json()["error"]["code"]) == (400, "unknown_mode")


def test_unknown_model_comes_before_unknown_mode(client, models):
    s = setup_project(client, n_images=1)
    r = client.post(f"/api/projects/{s['pid']}/jobs", json={"model": "nope", "mode": "nope"})
    assert r.json()["error"]["code"] == "unknown_model"


@pytest.mark.parametrize("body", [{}, {"mode": None}], ids=["absent", "null"])
def test_no_mode_uses_the_first_mode(client, models, body):
    s = setup_project(client, n_images=1)
    r = client.post(f"/api/projects/{s['pid']}/jobs", json={"model": "fake", **body})
    assert r.json()["config"]["mode"] == "text"


def test_no_mode_on_a_detector_without_modes_stores_null(tmp_path):
    with make_client(tmp_path, detectors=("yolo",), modes={"yolo": []})[0] as c:
        s = setup_project(c, n_images=1)
        r = c.post(f"/api/projects/{s['pid']}/jobs", json={"model": "yolo"})
        assert r.status_code == 202 and r.json()["config"]["mode"] is None


def test_default_mode_can_be_examples(tmp_path):
    with make_client(tmp_path, detectors=("yoloe",), modes={"yoloe": ["examples"]})[0] as c:
        s = setup_project(c, n_images=1)
        r = c.post(f"/api/projects/{s['pid']}/jobs", json={"model": "yoloe"})
        assert r.json()["config"]["mode"] == "examples"


def test_config_stores_the_given_mode(client, models):
    s = setup_project(client, n_images=1)
    for mode in ("text", "examples"):
        job = _job(client, s["pid"], mode=mode)
        assert job["config"] == {"model": "fake", "mode": mode, "image_ids": None, "options": {}}
        assert job["result"] is None


@pytest.mark.parametrize("config_extra", [{"mode": "text"}, {"mode": None}, {}],
                         ids=["text", "null", "old_job_without_mode"])
def test_text_jobs_call_run_as_before(client, models, config_extra):
    s = setup_project(client, n_images=1)
    iid = s["image_ids"][0]
    conn = db.connect(client.app.state.db_path)
    try:
        config = {"model": "fake", "image_ids": None, "options": {"x": 1}, **config_extra}
        db.create_job(conn, s["pid"], "detect", config, 1)
    finally:
        conn.close()
    assert client.app.state.worker.run_once()
    call = models.run_calls[0]
    assert call["kwargs"] == {"detector": "fake", "options": {"x": 1}}
    assert [k.name for k in call["classes"]] == ["person", "rb", "sb", "biax"]
    assert models.prepare_calls == []
    assert len(_rows(client, iid)) == 4


def test_text_job_has_null_result(client, models):
    s = setup_project(client, n_images=1)
    job = _job(client, s["pid"], mode="text")
    assert client.app.state.worker.run_once()
    assert client.get(f"/api/jobs/{job['id']}").json()["result"] is None


# ------------------------------------------------------------------ job flow and result

def test_prepare_and_run_get_the_right_arguments(client, models):
    s = setup_project(client, n_images=1)
    iid = s["image_ids"][0]
    _insert(client, iid, s["class_ids"]["sb"])
    _insert(client, iid, s["class_ids"]["rb"])
    _run(client, s["pid"], options={"x": 1})
    prep = models.prepare_calls[0]
    assert prep["detector"] == "fake" and prep["options"] == {"x": 1}
    assert [k.name for k in prep["classes"]] == ["rb", "sb"]  # project order
    assert set(prep["examples"]) == {"rb", "sb"}
    call = models.run_calls[0]
    assert [k.name for k in call["classes"]] == ["rb", "sb"]
    assert call["kwargs"]["detector"] == "fake" and call["kwargs"]["options"] == {"x": 1}
    assert [k.name for k in call["kwargs"]["example_run"].classes] == ["rb", "sb"]


def test_result_has_counts_and_both_skip_reasons(client, models):
    models.unusable = ("rb",)
    s = setup_project(client, n_images=1)
    iid, ids = s["image_ids"][0], s["class_ids"]
    for _ in range(3):
        _insert(client, iid, ids["person"])
    _insert(client, iid, ids["rb"])
    _insert(client, iid, ids["biax"])
    job = _run(client, s["pid"])
    assert job["status"] == "done" and job["error"] is None
    assert job["result"] == {
        "mode": "examples",
        "examples_by_class": {"person": 3, "biax": 1},
        "skipped_classes": [{"class_name": "sb", "reason": "no examples"},
                            {"class_name": "rb", "reason": "no usable examples"}]}
    assert [k.name for k in models.run_calls[0]["classes"]] == ["person", "biax"]


def test_reason_comes_from_the_detectors_skip_reasons(client, models):
    models.min_examples = 5
    models.unusable = ("rb",)  # rejected with no reason given -> the default reason
    s = setup_project(client, n_images=1)
    iid, ids = s["image_ids"][0], s["class_ids"]
    for _ in range(5):
        _insert(client, iid, ids["person"])
        _insert(client, iid, ids["rb"])
    for _ in range(4):
        _insert(client, iid, ids["biax"])
    job = _run(client, s["pid"])
    assert job["status"] == "done" and job["error"] is None
    status = client.get(f"/api/jobs/{job['id']}").json()  # the #10 job status endpoint
    assert status["result"]["examples_by_class"] == {"person": 5}
    assert status["result"]["skipped_classes"] == [
        {"class_name": "sb", "reason": "no examples"},
        {"class_name": "rb", "reason": "no usable examples"},
        {"class_name": "biax", "reason": "needs at least 5 examples"}]
    assert [k.name for k in models.run_calls[0]["classes"]] == ["person"]


def test_no_skips_with_skip_reasons_gives_an_empty_list(client, models):
    models.min_examples = 5
    s = setup_project(client, n_images=1)
    for cid in s["class_ids"].values():
        for _ in range(5):
            _insert(client, s["image_ids"][0], cid)
    job = _run(client, s["pid"])
    assert client.get(f"/api/jobs/{job['id']}").json()["result"]["skipped_classes"] == []


def test_skipped_classes_is_empty_when_nothing_was_skipped(client, models):
    s = setup_project(client, n_images=1)
    for cid in s["class_ids"].values():
        _insert(client, s["image_ids"][0], cid)
    assert _run(client, s["pid"])["result"]["skipped_classes"] == []


def test_result_is_written_before_images_run_at_progress_zero(client, models):
    s = setup_project(client, n_images=1)
    _insert(client, s["image_ids"][0], s["class_ids"]["rb"])
    seen = {}

    def look():
        conn = db.connect(client.app.state.db_path)
        try:
            seen.update(dict(conn.execute("SELECT status, progress, result FROM jobs").fetchone()))
        finally:
            conn.close()

    models.on_run = look
    _run(client, s["pid"])
    assert (seen["status"], seen["progress"]) == ("running", 0)
    assert '"examples_by_class": {"rb": 1}' in seen["result"]


def test_job_matches_the_fixture_keys(client, models):
    models.unusable = ("rb",)
    s = setup_project(client, n_images=1)
    iid, ids = s["image_ids"][0], s["class_ids"]
    for name in ("person", "rb", "biax"):
        _insert(client, iid, ids[name])
    same_keys(_run(client, s["pid"]), FIXTURES["GET /api/jobs/10 (detect, examples)"])


def test_prepare_failure_fails_the_job(client, models):
    models.prepare_error = OSError("cannot read example image")
    s = setup_project(client, n_images=1)
    iid = s["image_ids"][0]
    _insert(client, iid, s["class_ids"]["rb"])
    old = _insert(client, iid, s["class_ids"]["sb"], "needs_review", "fake", bbox=(1, 2, 3, 4))
    job = _run(client, s["pid"])
    assert job["status"] == "failed" and "cannot read example image" in job["error"]
    assert models.run_calls == []
    assert old in [a["id"] for a in _rows(client, iid)]


def test_box_for_a_skipped_class_fails_the_job(client, models):
    models.extra = [("sb", (1, 1, 5, 5), 0.9, "x")]  # sb has no examples, so it did not run
    s = setup_project(client, n_images=1)
    _insert(client, s["image_ids"][0], s["class_ids"]["rb"])
    job = _run(client, s["pid"])
    assert job["status"] == "failed" and "sb" in job["error"]


# ------------------------------------------------------------------ delete only classes that ran

def test_delete_is_limited_to_classes_that_ran(client, models):
    s = setup_project(client, n_images=1)
    iid, ids = s["image_ids"][0], s["class_ids"]
    _insert(client, iid, ids["rb"])  # the example: rb runs, sb has none
    old_rb = _insert(client, iid, ids["rb"], "needs_review", "owlv2", bbox=(1, 2, 3, 4))
    old_rb_auto = _insert(client, iid, ids["rb"], "auto_accepted", "owlv2", bbox=(50, 2, 3, 4))
    old_sb = _insert(client, iid, ids["sb"], "needs_review", "owlv2", bbox=(9, 9, 9, 9))
    old_sb_auto = _insert(client, iid, ids["sb"], "auto_accepted", "owlv2", bbox=(70, 9, 9, 9))
    job = _run(client, s["pid"])
    rows = {a["id"]: a for a in _rows(client, iid)}
    assert old_rb not in rows and old_rb_auto not in rows
    assert old_sb in rows and old_sb_auto in rows
    assert sorted(a["class_name"] for a in rows.values() if a["job_id"] == job["id"]) == ["rb"]


def test_detector_rejected_class_also_keeps_its_old_boxes(client, models):
    models.unusable = ("rb",)
    s = setup_project(client, n_images=1)
    iid, ids = s["image_ids"][0], s["class_ids"]
    _insert(client, iid, ids["rb"])
    _insert(client, iid, ids["sb"])
    _insert(client, iid, ids["rb"], "needs_review", "owlv2", bbox=(1, 2, 3, 4))
    _insert(client, iid, ids["sb"], "needs_review", "owlv2", bbox=(9, 9, 9, 9))
    _run(client, s["pid"])
    left = [a["bbox"] for a in _rows(client, iid)]  # by box: SQLite reuses a freed top id
    assert [1.0, 2.0, 3.0, 4.0] in left and [9.0, 9.0, 9.0, 9.0] not in left


def test_human_import_and_reviewed_boxes_of_a_class_that_ran_are_kept(client, models):
    s = setup_project(client, n_images=1)
    iid, cid = s["image_ids"][0], s["class_ids"]["rb"]
    keep = [_insert(client, iid, cid),
            _insert(client, iid, cid, "needs_review", "import", bbox=(1, 2, 3, 4)),
            _insert(client, iid, cid, "edited", "owlv2", bbox=(50, 2, 3, 4)),
            _insert(client, iid, cid, "rejected", "owlv2", bbox=(90, 2, 3, 4))]
    _run(client, s["pid"])
    assert set(keep) <= {a["id"] for a in _rows(client, iid)}


def test_overlap_rule_applies_in_examples_mode(client, models):
    s = setup_project(client, n_images=1)
    iid, cid = s["image_ids"][0], s["class_ids"]["rb"]
    _insert(client, iid, cid, bbox=(600, 300, 40, 40))  # same place as the model's new box
    job = _run(client, s["pid"])
    assert [a for a in _rows(client, iid) if a["job_id"] == job["id"]] == []


# ------------------------------------------------------------------ no class left

def test_no_class_left_is_done_at_once(client, models):
    s = setup_project(client, n_images=2)
    iid = s["image_ids"][0]
    old = _insert(client, iid, s["class_ids"]["rb"], "needs_review", "owlv2")
    job = _run(client, s["pid"])
    assert (job["status"], job["done"], job["total"], job["progress"]) == ("done", 2, 2, 1.0)
    assert job["finished_at"]
    assert job["result"]["examples_by_class"] == {}
    assert [x["class_name"] for x in job["result"]["skipped_classes"]] == \
        ["person", "rb", "sb", "biax"]
    assert models.prepare_calls == [] and models.run_calls == []
    assert [a["id"] for a in _rows(client, iid)] == [old]


def test_detector_rejecting_every_class_is_no_class_left(client, models):
    models.unusable = ("rb",)
    s = setup_project(client, n_images=1)
    iid = s["image_ids"][0]
    _insert(client, iid, s["class_ids"]["rb"])
    old = _insert(client, iid, s["class_ids"]["rb"], "needs_review", "owlv2", bbox=(1, 2, 3, 4))
    job = _run(client, s["pid"])
    assert (job["status"], job["done"], job["total"]) == ("done", 1, 1)
    assert {"class_name": "rb", "reason": "no usable examples"} in job["result"]["skipped_classes"]
    assert models.run_calls == []
    assert old in [a["id"] for a in _rows(client, iid)]


# ------------------------------------------------------------------ polygon from Prediction

def test_prediction_polygon_is_stored(client, models):
    models.polygon = TRIANGLE
    s = setup_project(client, n_images=1)
    iid = s["image_ids"][0]
    _insert(client, iid, s["class_ids"]["rb"], bbox=(1, 2, 3, 4))
    job = _run(client, s["pid"])
    new = next(a for a in _rows(client, iid) if a["job_id"] == job["id"])
    assert new["polygon"] == TRIANGLE and new["source"] == "new_model" and new["mask"] is None


def test_prediction_without_polygon_stores_null(client, models):
    s = setup_project(client, n_images=1)
    iid = s["image_ids"][0]
    _insert(client, iid, s["class_ids"]["rb"], bbox=(1, 2, 3, 4))
    job = _run(client, s["pid"])
    assert next(a for a in _rows(client, iid) if a["job_id"] == job["id"])["polygon"] is None


def test_polygon_is_stored_in_text_mode_too(client, models):
    models.polygon = TRIANGLE
    s = setup_project(client, n_images=1)
    _job(client, s["pid"], mode="text")
    assert client.app.state.worker.run_once()
    assert all(a["polygon"] == TRIANGLE for a in _rows(client, s["image_ids"][0]))
