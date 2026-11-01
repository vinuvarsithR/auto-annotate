"""
Shared helpers for the backend tests (no tests in here).

FakeClassSpec / FakePrediction stand in for autolabel.models.base until the
models branch is merged. fake_run behaves like the contract's "fake" detector:
2 boxes per image, for the first two classes, scores 0.9 and 0.3.
"""

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass

from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

from app.main import create_app

CRANE_CLASSES = [
    {"name": "person", "description": "a worker wearing a hard hat, seen from above",
     "min_score": 0.05, "accept_score": 0.5},
    {"name": "rb", "description": "a long narrow beige slab hanging from crane cables",
     "min_score": 0.05},
    {"name": "sb", "description": "a long thin yellow steel lifting beam", "min_score": 0.05},
    {"name": "biax", "description": "a short yellow crossbar hanging from crane cables",
     "min_score": 0.05},
]


@dataclass(frozen=True)
class FakeClassSpec:
    name: str
    description: str
    min_score: float | None = None


@dataclass(frozen=True)
class FakePrediction:
    image_id: str
    category: str
    bbox: tuple
    score: float = 1.0
    source: str = ""
    mask: dict | None = None


class FakePipeline:
    """Records calls; yields 2 boxes per image (scores 0.9 and 0.3)."""

    def __init__(self, fail_after: int | None = None, extra: list | None = None):
        self.calls = []
        self.fail_after = fail_after
        self.extra = extra or []

    def run(self, images, classes, detector="pipeline", options=None, on_progress=None):
        images = list(images)
        self.calls.append({"images": images, "classes": classes, "detector": detector,
                           "options": options})
        for n, (image_id, _path) in enumerate(images):
            if self.fail_after is not None and n >= self.fail_after:
                raise RuntimeError("CUDA out of memory while loading owlv2\nsecond line")
            preds = [
                FakePrediction(image_id, classes[0].name, (10, 20, 30, 40), 0.9, "fake"),
                FakePrediction(image_id, classes[1].name, (50, 60, 70, 80), 0.3, "fake"),
            ] + [FakePrediction(image_id, *e) for e in self.extra]
            yield image_id, preds

    def loader(self):
        return self.run, FakeClassSpec


def make_client(tmp_path, pipeline: FakePipeline | None = None,
                detectors=("pipeline", "fake")) -> tuple[TestClient, FakePipeline]:
    pipeline = pipeline or FakePipeline()
    app = create_app(tmp_path / "data", start_worker=False, load_pipeline=pipeline.loader,
                     list_detectors=lambda: list(detectors), static_dir=None)
    return TestClient(app), pipeline


_colour_counter = itertools.count()


def image_bytes(fmt: str = "JPEG", size=(1280, 720), colour=None) -> bytes:
    """A small solid image. Each call gets a new colour (so new bytes, not a #4 duplicate)
    unless `colour` is given; pass the same colour twice to get identical bytes."""
    if colour is None:
        n = next(_colour_counter)
        colour = (n % 256, (n // 256) % 256, 200)
    # JPEG can encode nearby colours to the same bytes, so the colour also goes into metadata.
    tag = f"test-image {colour}"
    buf = io.BytesIO()
    im = Image.new("RGB", size, colour)
    if fmt == "PNG":
        info = PngImagePlugin.PngInfo()
        info.add_text("tag", tag)
        im.save(buf, format=fmt, pnginfo=info)
    else:
        im.save(buf, format=fmt, comment=tag.encode())
    return buf.getvalue()


def setup_project(client: TestClient, n_images: int = 3, name: str = "eot_crane") -> dict:
    """Project + crane classes + n uploaded jpgs. Returns ids."""
    pid = client.post("/api/projects", json={"name": name}).json()["id"]
    files = [("files", (f"img_{i}.jpg", image_bytes(), "image/jpeg")) for i in range(n_images)]
    imgs = client.post(f"/api/projects/{pid}/images", files=files).json()["images"]
    cls = client.put(f"/api/projects/{pid}/classes", json={"classes": CRANE_CLASSES}).json()
    return {"pid": pid, "image_ids": [i["id"] for i in imgs],
            "class_ids": {c["name"]: c["id"] for c in cls["classes"]}}


def run_job(client: TestClient, pid: int, **body) -> dict:
    job = client.post(f"/api/projects/{pid}/jobs", json={"model": "pipeline", **body})
    assert job.status_code == 202, job.text
    assert client.app.state.worker.run_once()
    return client.get(f"/api/jobs/{job.json()['id']}").json()
