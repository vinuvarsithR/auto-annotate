"""Small check that autolabel/report.py writes one self-contained HTML file."""

from PIL import Image

from autolabel.report import build_report


def test_report_has_every_image_and_model_toggle(tmp_path):
    paths = []
    for n in ("a.jpg", "b.jpg"):
        Image.new("RGB", (200, 100), "gray").save(tmp_path / n)
        paths.append((n, tmp_path / n))
    out = tmp_path / "r.html"
    build_report(
        out, "T", ["PROVISIONAL labels"], "<table></table>", paths,
        gold={"a.jpg": [("sb", (10, 10, 50, 20))]},
        models={"yolo": {"a.jpg": [("sb", (12, 10, 50, 20), 0.87)]}, "gdino": {}},
        width=100,
    )
    page = out.read_text()
    assert page.count("data:image/jpeg;base64,") == 2  # images embedded, no outside files
    assert 'data-k="gold"' in page and 'data-k="m0"' in page and 'data-k="m1"' in page
    assert "sb 0.87" in page and "PROVISIONAL labels" in page
    assert 'x="5.0"' in page  # box scaled from 200 px to 100 px wide
