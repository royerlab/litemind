import os
from datetime import datetime
from pathlib import Path
from typing import List


def human_readable_size(size_in_bytes: int) -> str:
    """
    Converts a file size in bytes to a human-readable string
    with spelled-out units (bytes, kilobytes, megabytes, etc.).
    """
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} kilobytes"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / 1024 ** 2:.2f} megabytes"
    elif size_in_bytes < 1024**4:
        return f"{size_in_bytes / 1024 ** 3:.2f} gigabytes"
    elif size_in_bytes < 1024**5:
        return f"{size_in_bytes / 1024 ** 4:.2f} terabytes"
    else:
        return f"{size_in_bytes / 1024 ** 5:.2f} petabytes"


def format_datetime(timestamp: float) -> str:
    """
    Converts a timestamp (seconds from epoch) to a
    human-readable datetime string (YYYY-MM-DD HH:MM:SS).
    """
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def generate_tree_structure(
    folder_path: str,
    prefix: str = "",
    allowed_extensions: List[str] = None,
    excluded_files: List[str] = None,
    include_hidden_files=False,
):
    """
    Generates a directory tree *with file sizes* but WITHOUT timestamps.
    If you prefer not to display file sizes at all in the tree,
    you can remove the size-related lines below.
    """
    tree_str = ""

    contents = list(Path(folder_path).iterdir())
    # Optional: sort for consistency
    # contents.sort(key=lambda p: (p.is_file(), p.name.lower()))

    pointers = ["├── "] * (len(contents) - 1) + ["└── "]
    for pointer, path in zip(pointers, contents):
        # Skip hidden files and folders if include_hidden_files is False
        if not include_hidden_files and path.name.startswith("."):
            continue

        # Skip files in the excluded_files list
        if excluded_files and path.name in excluded_files:
            continue

        # Skip files with disallowed extensions
        if (
            allowed_extensions
            and path.is_file()
            and not any(path.name.endswith(ext) for ext in allowed_extensions)
        ):
            continue

        if path.is_dir():

            # Get the folder name without the path:
            folder_name = os.path.basename(path)

            # Skip hidden files if include_hidden_files is False
            if not include_hidden_files and (
                folder_name.startswith(".") or folder_name.startswith("__")
            ):
                continue

            # Folders: we won't show timestamps or a cumulative size here,
            # but you could if you really wanted to.
            tree_str += prefix + pointer + f"{path.name}/\n"
            extension = "│   " if pointer == "├── " else "    "

            tree_str += generate_tree_structure(
                folder_path=path,
                prefix=prefix + extension,
                allowed_extensions=allowed_extensions,
                excluded_files=excluded_files,
                include_hidden_files=include_hidden_files,
            )
        else:
            # Files: show only size in the tree
            size_str = human_readable_size(path.stat().st_size)
            tree_str += prefix + pointer + f"{path.name} ({size_str})\n"
    return tree_str


def is_text_file(file_path, blocksize=1024):
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(blocksize)
            # If the chunk contains null bytes, it's likely binary.
            if b"\0" in chunk:
                return False
            import chardet

            result = chardet.detect(chunk)
            # Check that an encoding was found and that the confidence is reasonably high.
            if not result["encoding"] or result["confidence"] < 0.5:
                return False
            return True
    except Exception:
        return False


def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_binary_file_info(file_path):
    with open(file_path, "rb") as f:
        content = f.read(100)
        return len(content), content.hex()


def file_info_header(file_path: str, file_type_label: str) -> str:
    """
    Build a header string with size, timestamps, and file type label.
    """
    stat_info = os.stat(file_path)
    size_str = human_readable_size(stat_info.st_size)
    mod_str = format_datetime(stat_info.st_mtime)
    cre_str = format_datetime(
        stat_info.st_ctime
    )  # On Unix, metadata change; on Windows, creation
    filename = os.path.basename(file_path)

    return (
        f"\n================================================\n"
        f"{file_type_label} File: {filename}\n"
        f"Size: {size_str}\n"
        f"Last Modified: {mod_str}\n"
        f"Created: {cre_str}\n"
        f"================================================\n"
    )
