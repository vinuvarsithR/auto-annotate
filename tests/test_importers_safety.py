"""check_names / check_zip / extract_zip (CONTRACT #22)."""

import io
import warnings
import zipfile

import pytest
from PIL import Image

from autolabel.importers import IMAGE_EXTS, ImportRejected, check_names, check_zip, extract_zip


def png_bytes(w=8, h=6):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, "PNG")
    return buf.getvalue()


def make_zip(path, entries):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return path


def test_image_exts():
    assert IMAGE_EXTS == {".jpg", ".jpeg", ".png"}


@pytest.mark.parametrize("names", [
    ["a.jpg", "train/images/b.png", "x/./y.jpg", "a..b.jpg", "train\\labels\\b.txt"],
    [],
])
def test_check_names_accepts_safe(names):
    check_names(names)


@pytest.mark.parametrize("bad", [
    "../a.jpg", "x/../../a.jpg", "x/..", "..\\a.jpg", "x\\..\\a.jpg",
    "/etc/passwd", "\\x.jpg", "C:\\x.jpg", "c:/x.jpg", "", "/",
])
def test_check_names_rejects_unsafe(bad):
    with pytest.raises(ImportRejected):
        check_names(["ok.jpg", bad])


def test_check_names_rejects_repeated_path():
    with pytest.raises(ImportRejected, match="twice"):
        check_names(["train/a.jpg", "valid/a.jpg", "train/a.jpg"])
    with pytest.raises(ImportRejected, match="twice"):
        check_names(["train/a.jpg", "train\\a.jpg"])  # same path, other separator


def test_import_rejected_is_value_error():
    assert issubclass(ImportRejected, ValueError)


def test_check_zip_counts_images(tmp_path):
    z = make_zip(tmp_path / "u.zip", {
        "train/images/a.jpg": b"x", "train/images/B.PNG": b"x", "c.jpeg": b"x",
        "train/labels/a.txt": b"", "data.yaml": b"names: [a]", "notes/": b"",
    })
    assert check_zip(z, 10**9) == 3


def test_check_zip_not_a_zip(tmp_path):
    p = tmp_path / "u.zip"
    p.write_bytes(b"not a zip at all")
    with pytest.raises(ImportRejected, match="zip"):
        check_zip(p, 10**9)


def test_check_zip_missing_file(tmp_path):
    with pytest.raises(ImportRejected):
        check_zip(tmp_path / "nope.zip", 10**9)


def test_check_zip_unsafe_entry(tmp_path):
    z = make_zip(tmp_path / "u.zip", {"ok.jpg": b"x", "../evil.jpg": b"x"})
    with pytest.raises(ImportRejected, match=r"\.\."):
        check_zip(z, 10**9)


def test_check_zip_absolute_entry(tmp_path):
    z = make_zip(tmp_path / "u.zip", {"/abs.jpg": b"x"})
    # zipfile strips nothing on write; the stored name keeps the leading slash
    with pytest.raises(ImportRejected):
        check_zip(z, 10**9)


def test_check_zip_over_cap(tmp_path):
    z = make_zip(tmp_path / "u.zip", {"a.jpg": b"x" * 600, "b.jpg": b"x" * 600})
    assert check_zip(z, 1200) == 2
    with pytest.raises(ImportRejected, match="too large"):
        check_zip(z, 1199)


def test_extract_zip_unpacks_under_dest(tmp_path):
    data = png_bytes()
    z = make_zip(tmp_path / "u.zip", {"train/images/a.png": data, "README.txt": b"hi", "empty/": b""})
    dest = tmp_path / "out"
    assert extract_zip(z, dest, 10**9) == []
    assert (dest / "train/images/a.png").read_bytes() == data
    assert (dest / "README.txt").read_bytes() == b"hi"


def test_extract_zip_rejects_unsafe(tmp_path):
    z = make_zip(tmp_path / "u.zip", {"../evil.txt": b"x"})
    with pytest.raises(ImportRejected):
        extract_zip(z, tmp_path / "out", 10**9)
    assert not (tmp_path / "evil.txt").exists()


def test_extract_zip_stops_file_past_cap(tmp_path):
    z = make_zip(tmp_path / "u.zip", {"a.jpg": b"a" * 100, "b.jpg": b"b" * 100, "c.jpg": b"c" * 10})
    dest = tmp_path / "out"
    skipped = extract_zip(z, dest, 150)
    assert skipped == [{"file_name": "b.jpg", "reason": "too large"}]
    assert (dest / "a.jpg").exists() and not (dest / "b.jpg").exists()
    assert (dest / "c.jpg").read_bytes() == b"c" * 10


def test_extract_zip_corrupt_entry_is_skipped(tmp_path):
    z = tmp_path / "u.zip"
    with zipfile.ZipFile(z, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("a.jpg", b"a" * 2000)
        zf.writestr("b.jpg", b"b" * 50)
    raw = bytearray(z.read_bytes())
    # flip bytes in a.jpg's compressed data (right after its 30-byte header + name)
    start = 30 + len("a.jpg")
    for i in range(start, start + 6):
        raw[i] ^= 0xFF
    z.write_bytes(bytes(raw))
    skipped = extract_zip(z, tmp_path / "out", 10**9)
    assert skipped == [{"file_name": "a.jpg", "reason": "cannot unpack"}]
    assert (tmp_path / "out/b.jpg").read_bytes() == b"b" * 50


# ---------- repeated zip entries ----------

def make_zip_with_repeats(path, entries):
    """entries: (name, data) pairs; zipfile warns about a repeated name but writes it."""
    with zipfile.ZipFile(path, "w") as zf, warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        for name, data in entries:
            zf.writestr(name, data)
    return path


@pytest.mark.parametrize("second", [
    b"names: [a, b]",   # different size
    b"names: [b]",      # same size, different CRC
])
def test_zip_repeated_entry_with_different_content_is_rejected(tmp_path, second):
    z = make_zip_with_repeats(tmp_path / "u.zip", [
        ("data.yaml", b"names: [a]"), ("a.png", png_bytes()), ("data.yaml", second)])
    with pytest.raises(ImportRejected) as check_err:
        check_zip(z, 10**9)
    assert str(check_err.value) == "path appears twice: data.yaml"
    with pytest.raises(ImportRejected) as extract_err:
        extract_zip(z, tmp_path / "out", 10**9)
    assert str(extract_err.value) == "path appears twice: data.yaml"


def test_zip_identical_repeats_count_once_toward_size_cap(tmp_path):
    z = make_zip_with_repeats(tmp_path / "u.zip", [("a.jpg", b"x" * 600)] * 3)
    assert check_zip(z, 600) == 1
    with pytest.raises(ImportRejected, match="too large"):
        check_zip(z, 599)
    dest = tmp_path / "out"
    assert extract_zip(z, dest, 600) == []
    assert (dest / "a.jpg").read_bytes() == b"x" * 600
