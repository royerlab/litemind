"""Utilities for generating human-readable folder descriptions and tree structures."""

import fnmatch
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def human_readable_size(size_in_bytes: int) -> str:
    """
    Convert a file size in bytes to a human-readable string.

    Parameters
    ----------
    size_in_bytes : int
        The file size in bytes.

    Returns
    -------
    str
        A human-readable size string with spelled-out units
        (e.g., ``"1.50 megabytes"``).
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
    Convert a Unix timestamp to a human-readable datetime string.

    Parameters
    ----------
    timestamp : float
        Seconds since the Unix epoch.

    Returns
    -------
    str
        Formatted datetime string in ``"YYYY-MM-DD HH:MM:SS"`` format.
    """
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


def generate_tree_structure(
    folder_path: str,
    prefix: str = "",
    allowed_extensions: List[str] = None,
    excluded_files: List[str] = None,
    include_hidden_files=False,
    depth: Optional[int] = None,  # None means unlimited depth
):
    """
    Generate a tree structure string representation of a folder's contents.

    Recursively traverses the folder and its subfolders, filtering files
    based on allowed extensions, excluded files, and whether to include
    hidden files or not. Respects a depth limit for recursion, where
    None means unlimited depth.

    Parameters
    ----------
    folder_path : str
        The path to the folder to traverse.
    prefix : str
        The prefix to use for each line in the tree structure.
    allowed_extensions : list of str, optional
        A list of allowed file extensions. If None, all files are included.
    excluded_files : list of str, optional
        A list of file name patterns to exclude from the tree structure.
    include_hidden_files : bool, optional
        Whether to include hidden files (starting with a dot). Default
        is False.
    depth : int or None, optional
        The maximum depth to traverse. If None, there is no limit.
        If set to 0, only the top-level folder is shown.

    Returns
    -------
    str
        A string representing the tree structure of the folder contents.
    """
    tree_str = ""

    # If depth is None, set it to a very high number to allow unlimited depth
    if depth is None:
        depth = -1  # Use -1 to indicate unlimited depth

    # Return if maximum depth is reached, but only if it's specifically set to 0
    if depth == 0:
        return tree_str + prefix + "... (further contents truncated)\n"

    contents = list(Path(folder_path).iterdir())
    # Sort for consistency: files first, then directories, both alphabetically
    contents.sort(key=lambda p: (p.is_dir(), p.name.lower()))

    # Filter contents based on criteria
    filtered_contents = []
    for path in contents:
        # Skip hidden files and folders if include_hidden_files is False
        if not include_hidden_files and path.name.startswith("."):
            continue

        # Skip files in the excluded_files list
        if excluded_files and any(
            fnmatch.fnmatch(path.name, p) for p in excluded_files
        ):
            continue

        # Skip files with disallowed extensions
        if (
            allowed_extensions
            and path.is_file()
            and not any(path.name.endswith(ext) for ext in allowed_extensions)
        ):
            continue

        # Skip hidden directories with double underscores if include_hidden_files is False
        if not include_hidden_files and path.is_dir() and path.name.startswith("__"):
            continue

        filtered_contents.append(path)

    pointers = (
        ["├── "] * (len(filtered_contents) - 1) + ["└── "] if filtered_contents else []
    )

    for pointer, path in zip(pointers, filtered_contents):
        if path.is_dir():
            # Display the folder
            tree_str += prefix + pointer + f"{path.name}/\n"
            extension = "│   " if pointer == "├── " else "    "

            # Recurse with decremented depth
            next_depth = depth - 1 if depth > 0 else depth
            tree_str += generate_tree_structure(
                folder_path=path,
                prefix=prefix + extension,
                allowed_extensions=allowed_extensions,
                excluded_files=excluded_files,
                include_hidden_files=include_hidden_files,
                depth=next_depth,
            )
        else:
            # Files: show only size in the tree
            size_str = human_readable_size(path.stat().st_size)
            tree_str += prefix + pointer + f"{path.name} ({size_str})\n"

    return tree_str


def read_file_content(file_path: str) -> str:
    """
    Read and return the text content of a file.

    Parameters
    ----------
    file_path : str
        Path to the file to read.

    Returns
    -------
    str
        The file content as a string, with undecodable characters ignored.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_binary_file_info(file_path: str):
    """
    Read the first 100 bytes of a binary file.

    Parameters
    ----------
    file_path : str
        Path to the binary file.

    Returns
    -------
    tuple of (int, str)
        A tuple of the number of bytes read and their hex representation.
    """
    with open(file_path, "rb") as f:
        content = f.read(100)
        return len(content), content.hex()


def file_info_header(
    file_path: str,
    file_type_label: str,
    date_and_times: bool = True,
    file_sizes: bool = True,
) -> str:
    """
    Build a header string with size, timestamps, and file type label.

    Parameters
    ----------
    file_path : str
        The path to the file.
    file_type_label : str
        A label for the type of file (e.g., ``"Text"``, ``"Binary"``).
    date_and_times : bool, optional
        Whether to include modification and creation dates in the header.
        Default is True.
    file_sizes : bool, optional
        Whether to include file size in the header. Default is True.

    Returns
    -------
    str
        A formatted string containing file metadata.
    """
    stat_info = os.stat(file_path)
    filename = os.path.basename(file_path)

    # Build header content parts
    header_parts = [f"{file_type_label} File: {filename}"]

    # Add size information if requested
    if file_sizes:
        size_str = human_readable_size(stat_info.st_size)
        header_parts.append(f"Size: {size_str}")

    # Add date/time information if requested
    if date_and_times:
        mod_str = format_datetime(stat_info.st_mtime)
        # On Unix, st_ctime is metadata change time; on Windows, creation time
        cre_str = format_datetime(stat_info.st_ctime)
        header_parts.append(f"Last Modified: {mod_str}")
        header_parts.append(f"Created: {cre_str}")

    # Build the complete header with consistent borders
    border = "=" * 34
    header = f"\n{border}\n" + "\n".join(header_parts) + f"\n{border}\n"

    return header
