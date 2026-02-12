"""
Shared utilities for the litemind CLI commands.

Provides helper functions used by multiple CLI subcommands, such as
default folder-scanning parameters for repository export.
"""

from typing import List, Optional, Tuple


def default_folder_scanning_parameters(
    allowed_extensions: Optional[List[str]],
    excluded_files: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Return folder scanning parameters, filling in sensible defaults.

    When either argument is ``None`` the function supplies a built-in
    default list.  Default extensions include ``.py``, ``.md``, ``.txt``,
    ``.toml``, ``LICENSE``, ``.tests``, and ``.html``.  Default exclusions
    cover ``litemind.egg-info``, ``dist``, and ``build`` directories.

    Parameters
    ----------
    allowed_extensions : Optional[List[str]]
        File extensions to include in scanning. If ``None``, the built-in
        default list is used.
    excluded_files : Optional[List[str]]
        File or directory names to exclude. If ``None``, the built-in
        default list is used.

    Returns
    -------
    Tuple[List[str], List[str]]
        A ``(allowed_extensions, excluded_files)`` tuple with defaults
        applied where the caller passed ``None``.
    """
    # Set default values for allowed_extensions and excluded_files:
    if allowed_extensions is None:
        allowed_extensions = [
            ".py",
            ".md",
            ".txt",
            ".toml",
            "LICENSE",
            ".tests",
            ".html",
        ]
    # Set default values for excluded_files:
    if excluded_files is None:
        excluded_files = [
            "litemind.egg-info",
            "dist",
            "build",
        ]
    return allowed_extensions, excluded_files
