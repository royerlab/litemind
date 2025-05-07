"""
File-type detection helpers
---------------------------
Dependencies   : pip install python-magic filetype
Windows users  : if libmagic is awkward to install, use
                 pip install python-magic-bin  # includes a DLL
"""

from __future__ import annotations

import mimetypes
import os
from typing import Any, Dict

# 1) Fast, pure-Python signature matcher
import filetype  # pip install filetype                ──┐

# ↳ supports >30 common kinds             │
# 2) Deep libmagic lookup (optional but richer)                       │
try:  # │
    import magic  # pip install python-magic ----------┘  │
except ImportError:  # │
    magic = None  # ┘


def probe(path: str, read_bytes: int = 4096) -> Dict[str, Any]:
    """
    Return a dict describing the file.
      • extension_mime  – MIME guessed from the file-name
      • signature_mime  – MIME guessed from magic numbers (filetype)
      • libmagic_mime   – MIME guessed by libmagic (if available)
      • is_text         – True/False based on null-byte & UTF-8 decode
    """
    info: Dict[str, Any] = {}

    # --- 1. Extension-based guess ---------------------------------
    mime_ext, _ = mimetypes.guess_type(path, strict=False)
    info["extension_mime"] = mime_ext

    # --- 2. Signature-based guess (file header) -------------------
    try:
        kind = filetype.guess(path)
        info["signature_mime"] = kind.mime if kind else None
    except Exception:
        info["signature_mime"] = None

    # --- 3. libmagic (deeper database) ----------------------------
    if magic:
        try:
            m = magic.Magic(mime=True)
            info["libmagic_mime"] = m.from_file(path)
        except Exception:
            info["libmagic_mime"] = None
    else:
        info["libmagic_mime"] = None

    # --- 4. Text vs binary heuristic ------------------------------
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(read_bytes)
            if b"\x00" in chunk:
                info["is_text"] = False
            else:
                try:
                    chunk.decode("utf-8")
                    info["is_text"] = True
                except UnicodeDecodeError:
                    info["is_text"] = False
    except Exception:
        info["is_text"] = None

    return info


# Core sets of MIME sub-types per category
_IMAGE = {"image/"}
_AUDIO = {"audio/"}
_VIDEO = {"video/"}
_ARCHIVE = {
    "application/zip",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/x-rar-compressed",
    "application/gzip",
    "application/x-bzip2",
}
_OFFICE = {
    # OOXML
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    # Binary legacy
    "application/msword",
    "application/vnd.ms-excel",
    "application/vnd.ms-powerpoint",
    # ODF
    "application/vnd.oasis.opendocument.text",
    "application/vnd.oasis.opendocument.spreadsheet",
}
_EXECUTABLE = {
    "application/vnd.microsoft.portable-executable",  # Windows PE:contentReference[oaicite:0]{index=0}
    "application/x-dosexec",  # libmagic for PE:contentReference[oaicite:1]{index=1}
    "application/x-executable",
    "application/x-elf",  # ELF:contentReference[oaicite:2]{index=2}
    "application/x-mach-binary",
}

_SCRIPT_EXTS = {".py", ".js", ".sh", ".bat", ".ps1", ".rb", ".pl"}


def classify(path: str, read_bytes: int = 4096) -> str:
    """
    Return one of:
      'text file' | 'script' | 'PDF document' | 'image' | 'audio' |
      'video' | 'archive' | 'office document' | 'executable' | 'binary file'
    """
    meta = probe(path, read_bytes)
    mime = (
        meta.get("libmagic_mime")
        or meta.get("signature_mime")
        or meta.get("extension_mime")
        or ""
    )

    # ---- 1. PDF: -----------------------------------
    if "pdf" in mime.lower():
        return "PDF document"

    # ---- 2. Text / script -------------------------------------------------
    if mime.startswith("text") or meta["is_text"]:
        if os.path.splitext(path)[1].lower() in _SCRIPT_EXTS:
            return "script"
        return "text file"

    # ---- 3. Media families ------------------------------------------------
    if any(
        mime.startswith(p) for p in _IMAGE
    ):  # JPEG/PNG/GIF … :contentReference[oaicite:4]{index=4}
        return "image"
    if any(
        mime.startswith(p) for p in _AUDIO
    ):  # MP3/WAV … :contentReference[oaicite:5]{index=5}
        return "audio"
    if any(
        mime.startswith(p) for p in _VIDEO
    ):  # MP4/WebM … :contentReference[oaicite:6]{index=6}
        return "video"

    # ---- 4. Archives ------------------------------------------------------
    if mime in _ARCHIVE:  # ZIP/TAR/7Z … :contentReference[oaicite:7]{index=7}
        return "archive"

    # ---- 5. Office docs ---------------------------------------------------
    if mime in _OFFICE:  # DOCX/XLSX/PPTX/ODT … :contentReference[oaicite:8]{index=8}
        return "office document"

    # ---- 6. Executables ---------------------------------------------------
    if mime in _EXECUTABLE:
        return "executable"

    # ---- 7. Fallback ------------------------------------------------------
    return "binary file"


def is_text_file(path: str, read_bytes: int = 4096) -> bool | None:
    """
    Convenience wrapper around probe().

    Returns
    -------
    bool | None
        • True   – looks like plain text (incl. UTF-8/ASCII/UTF-16 BOM etc.)
        • False  – has NUL bytes or other binary markers
        • None   – file unreadable or error occurred
    """
    meta = probe(path, read_bytes)
    return meta["is_text"]
