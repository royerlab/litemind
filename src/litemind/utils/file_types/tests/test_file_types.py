"""
test_file_probe.py – pytest checks for file_probe.probe() / classify()
---------------------------------------------------------------------
These tests create temporary files of several kinds and assert:

  • probe() returns the expected MIME hints and text/binary flag
  • classify() produces the right high-level label
  • Content sniffing overrides misleading extensions
"""

import base64
import io
import sys
import zipfile

import pytest

from litemind.ressources.media_resources import MediaResources
from litemind.utils.file_types.file_types import classify, is_text_file, probe
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


# --------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------
def _write_bytes(p, data: bytes) -> str:
    """Write *data* to tmp_path/file and return str path."""
    p.write_bytes(data)
    return str(p)


def _write_text(p, txt: str) -> str:
    p.write_text(txt, encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------
# Byte literals for test assets
# --------------------------------------------------------------------
# A valid, 1×1-pixel PNG (transparent) as base64:
_PNG_1X1 = base64.b64decode(
    b"iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8A"
    b"AwMB/6WADDsAAAAASUVORK5CYII="
)

# A **minimal** but valid PDF skeleton (adapted from Adobe spec examples)
_PDF_MIN = (
    b"%PDF-1.4\n"
    b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
    b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 1 1] >>\nendobj\n"
    b"xref\n0 4\n0000000000 65535 f \n"
    b"0000000010 00000 n \n0000000060 00000 n \n0000000117 00000 n \n"
    b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n176\n%%EOF\n"
)


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
def test_text_file_is_detected(tmp_path):
    p = tmp_path / "hello.txt"
    f = _write_text(p, "Hello world!\n")
    meta = probe(f)
    assert meta["is_text"] is True, meta
    assert classify(f) == "text"


def test_pdf_file_is_detected(tmp_path):
    p = tmp_path / "doc.pdf"
    f = _write_bytes(p, _PDF_MIN)
    assert classify(f) == "pdf"
    meta = probe(f)
    # At least one of the MIME avenues should say PDF
    assert (
        ("pdf" in (meta.get("extension_mime") or ""))
        or ("pdf" in (meta.get("signature_mime") or ""))
        or ("pdf" in (meta.get("libmagic_mime") or ""))
    )


def test_png_file_is_detected(tmp_path):
    p = tmp_path / "img.png"
    f = _write_bytes(p, _PNG_1X1)
    meta = probe(f)
    assert classify(f) == "image"
    assert (meta.get("signature_mime") or meta.get("extension_mime")).startswith(
        "image/"
    )


def test_binary_file_is_detected(tmp_path):
    p = tmp_path / "raw.xzyst"
    # Null-bytes will trigger the binary heuristic
    f = _write_bytes(p, b"\xff\xfe\xfd\xfc\x00\x01\x02\x03RandomBinary\x00\xff")
    meta = probe(f)
    assert meta["is_text"] is False
    assert classify(f) == "binary"


def test_content_over_extension_precedence(tmp_path):
    """
    If we rename a PNG with .txt, classify() should still say 'image'
    because the *content* wins over the misleading extension.
    """
    p = tmp_path / "picture.txt"
    f = _write_bytes(p, _PNG_1X1)
    assert classify(f) == "image"
    meta = probe(f)
    # extension_mime is text/plain (or None), but signature/libmagic says image/png
    assert meta["signature_mime"] == "image/png"


# --------------------------------------------------------------------
# Optional: tests that rely on python-magic only run if it's available
# --------------------------------------------------------------------
magic = sys.modules.get("magic")  # imported by file_probe at runtime


@pytest.mark.skipif(magic is None, reason="python-magic not installed")
def test_libmagic_integration(tmp_path):
    p = tmp_path / "sample.gif"
    # GIF 89a header:
    gif_bytes = (
        b"GIF89a\x01\x00\x01\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff!\xf9\x04"
        b"\x01\x00\x00\x00\x00,\x00\x00\x00\x00\x01\x00\x01\x00\x00\x02\x02"
        b"\x4c\x01\x00;"
    )
    f = _write_bytes(p, gif_bytes)
    meta = probe(f)
    assert meta["libmagic_mime"] == "image/gif"
    assert classify(f) == "image"


def test_code_detected(tmp_path):
    p = tmp_path / "hello.py"
    f = _write_text(p, "print('hi')\n")
    assert classify(f) == "code"


def test_zip_archive_detected(tmp_path):
    p = tmp_path / "data.zip"
    # Create an in-memory empty ZIP and write to disk
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w"):
        pass
    f = _write_bytes(p, zbuf.getvalue())
    assert classify(f) == "archive"


def test_docx_office_detected(tmp_path):
    # Use media_ressources.py to get the local test file URI:
    docx_path = MediaResources.get_local_test_document_uri("maya_takahashi_cv.docx")

    local_path = uri_to_local_file_path(docx_path)

    assert classify(local_path) == "office"


# --------------- Optional magics still here (unchanged) ------------------------
magic = sys.modules.get("magic")


@pytest.mark.skipif(magic is None, reason="python-magic not installed")
def test_libmagic_pe(tmp_path):
    p = tmp_path / "hello.exe"
    _write_bytes(p, b"MZFakePE")
    meta = probe(str(p))
    assert meta["libmagic_mime"] in {
        "application/x-dosexec",
        "application/vnd.microsoft.portable-executable",
    }


def test_is_text_file_true(tmp_path):
    f = (tmp_path / "note.txt").write_text("Hello!\n", encoding="utf-8")
    assert is_text_file(str(tmp_path / "note.txt")) is True


def test_is_text_file_false(tmp_path):
    raw = (tmp_path / "data.bin").write_bytes(b"\x00\xff\x00\xff")
    assert is_text_file(str(tmp_path / "data.bin")) is False
