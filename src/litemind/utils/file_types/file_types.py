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
from functools import lru_cache
from typing import Any, Dict, Set

# 1) Fast, pure-Python signature matcher
import filetype  # pip install filetype                ──┐

from litemind.utils.file_types.file_extensions import (
    ARCHIVE_EXTS,
    AUDIO_EXTS,
    EXECUTABLE_EXTS,
    IMAGE_EXTS,
    OFFICE_EXTS,
    OTHER_BINARY_EXTS,
    PROG_LANG_EXTS,
    SCRIPT_EXTS,
    VIDEO_EXTS,
    WEB_EXTS,
)

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
_TEXT = {"text/"}
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


def classify_uri(uri: str, read_bytes: int = 4096) -> str:
    """
    Convenience wrapper around classify().

    Returns one of:
      'text' | 'code' | 'pdf' | 'image' | 'audio' |
      'video' | 'archive' | 'office document' | 'executable' | 'web' | 'binary'
    """
    # Convert URI to local file path:
    from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path

    local_path = uri_to_local_file_path(uri)
    return classify(local_path, read_bytes)


@lru_cache()
def classify(path: str, read_bytes: int = 4096) -> str:
    """
    Return one of:
      'text' | 'code' | 'pdf' | 'image' | 'audio' |
      'video' | 'archive' | 'office' | 'executable' | 'web' | 'binary'
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
        return "pdf"

    # ---- 2. Text / script -------------------------------------------------
    if mime.startswith("text") or meta["is_text"]:
        if os.path.splitext(path)[1].lower() in SCRIPT_EXTS:
            return "script"
        if os.path.splitext(path)[1].lower() in PROG_LANG_EXTS:
            return "code"
        return "text"

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
        return "office"

    # ---- 6. Executables ---------------------------------------------------
    if mime in _EXECUTABLE:
        return "executable"

    if mime in WEB_EXTS:
        return "web"

    # ---- 7. Fallback: use extensions --------------------------------------
    ext = os.path.splitext(path)[1].lower()
    if ext in SCRIPT_EXTS:
        return "script"
    if ext in PROG_LANG_EXTS:
        return "code"
    if ext in [".pdf"]:
        return "pdf"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in ARCHIVE_EXTS:
        return "archive"
    if ext in OFFICE_EXTS:
        return "office"
    if ext in EXECUTABLE_EXTS:
        return "executable"
    if ext in OTHER_BINARY_EXTS:
        return "binary"
    if ext in WEB_EXTS:
        return "web"

    # ---- 8. Fallback ------------------------------------------------------
    return "binary"


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
    return classify(path) == "text"


def is_script_file(path: str, read_bytes: int = 4096) -> bool | None:
    """
    Check if the file is a script file based on its extension and content.

    Parameters
    ----------
    path : str
        The path to the file.
    read_bytes : int, optional
        The number of bytes to read from the file for probing, by default 4096.
    Returns
    -------
    bool | None
        • True   – looks like script file based on extension and content
        • False  – does not look like script file
    """
    return classify(path) == "script"


def is_prog_code_file(path: str, read_bytes: int = 4096) -> bool:
    """
    Check if the file is a programming language code file based on its extension and content.

    Parameters
    ----------
    path : str
        The path to the file.
    read_bytes : int, optional
        The number of bytes to read from the file for probing, by default 4096.
    Returns
    -------
    bool
        True if the file is a code file, False otherwise.
    """
    return classify(path) == "code"


def is_pdf_file(path: str) -> bool:
    """
    Check if the file is a PDF based on its MIME type.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    bool
        True if the file is a PDF, False otherwise.
    """
    return classify(path) == "pdf"


def is_image_file(path: str) -> bool:
    """
    Check if the file is an image based on its MIME type.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    bool
        True if the file is an image, False otherwise.
    """
    return classify(path) == "image"


def is_audio_file(path: str) -> bool:
    """
    Check if the file is an audio file based on its MIME type.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    bool
        True if the file is an audio file, False otherwise.
    """
    return classify(path) == "audio"


def is_video_file(path: str) -> bool:
    """
    Check if the file is a video file based on its MIME type.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    bool
        True if the file is a video file, False otherwise.
    """
    return classify(path) == "video"


def is_archive_file(path: str) -> bool:
    """
    Check if the file is an archive file based on its MIME type.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    bool
        True if the file is an archive file, False otherwise.
    """
    return classify(path) == "archive"


def is_office_file(path: str) -> bool:
    """
    Check if the file is an office document based on its MIME type.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    bool
        True if the file is an office document, False otherwise.
    """

    return classify(path) == "office"


def is_executable_file(path: str) -> bool:
    """
    Check if the file is an executable file based on its MIME type.
    Parameters
    ----------
    path

    Returns
    -------

    """
    return classify(path) == "executable"


def is_web_file(path: str) -> bool:
    """
    Check if the file is a web file based on its MIME type.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    bool
        True if the file is a web file, False otherwise.
    """
    return classify(path) == "web"


def is_document_file(path: str) -> bool:
    """
    Check if the file is a document file based on its MIME type.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    bool
        True if the file is a document file, False otherwise.
    """
    return (
        is_text_file(path)
        or is_office_file(path)
        or is_pdf_file(path)
        or is_prog_code_file(path)
        or is_script_file(path)
        or is_web_file(path)
    )


def get_extension(path: str) -> str:
    """
    Get the file extension from the path.

    Parameters
    ----------
    path : str
        The path to the file.

    Returns
    -------
    str
        The file extension, or an empty string if no extension is found.
    """
    return os.path.splitext(path)[1].lower() if os.path.splitext(path)[1] else ""


def has_extension(path: str, extensions: Set[str]) -> bool:
    """
    Check if the file has one of the specified extensions.

    Parameters
    ----------
    path : str
        The path to the file.
    extensions : Set[str]
        A list of file extensions to check against.

    Returns
    -------
    bool
        True if the file has one of the specified extensions, False otherwise.
    """
    return os.path.splitext(path)[1].lower() in [ext.lower() for ext in extensions]
