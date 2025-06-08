import fnmatch
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional


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
    depth: Optional[int] = None,  # None means unlimited depth
):
    """
    Generate a tree structure string representation of a folder's contents.
    This function recursively traverses the folder and its subfolders,
    filtering files based on allowed extensions, excluded files, and
    whether to include hidden files or not. It also respects a depth limit
    for recursion, where None means unlimited depth.

    Parameters
    ----------
    folder_path: str
        The path to the folder to traverse.
    prefix: str
        The prefix to use for each line in the tree structure.
    allowed_extensions: List[str]
        A list of allowed file extensions. If None, all files are included.
    excluded_files: List[str]
        A list of file names to exclude from the tree structure.
    include_hidden_files: bool
        Whether to include hidden files (those starting with a dot) in the tree structure.
    depth: Optional[int]
        The maximum depth to traverse. If None, there is no limit.
        If set to 0, it will only show the top-level folder and truncate further contents.

    Returns
    -------

    """
    tree_str = ""

    # If depth is None, set it to a very high number to allow unlimited depth
    if depth is None:
        depth = -1  # Use -1 to indicate unlimited depth

    # Return if maximum depth is reached, but only if it's specifically set to 0
    if depth == 0:
        return tree_str + prefix + "... (further contents truncated)\n"

    contents = list(Path(folder_path).iterdir())
    # Optional: sort for consistency
    # contents.sort(key=lambda p: (p.is_file(), p.name.lower()))

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
            # Get the folder name without the path
            folder_name = os.path.basename(path)

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


def read_file_content(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_binary_file_info(file_path):
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
    file_path: str
        The path to the file.
    file_type_label: str
        A label for the type of file (e.g., "Text", "Binary", etc.).
    date_and_times: bool
        Whether to include modification and creation dates in the header.
    file_sizes: bool
        Whether to include file size in the header.

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
