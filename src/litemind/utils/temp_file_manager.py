"""
Temporary file manager with automatic cleanup at process exit.

This module provides utilities for creating temporary files and directories
that are automatically cleaned up when the Python process exits.
"""

import atexit
import os
import shutil
from typing import Set

_temp_files: Set[str] = set()
_temp_dirs: Set[str] = set()


def register_temp_file(filepath: str) -> str:
    """
    Register a temporary file for cleanup at process exit.

    Parameters
    ----------
    filepath : str
        The path to the temporary file.

    Returns
    -------
    str
        The same filepath (for convenience in chaining).
    """
    _temp_files.add(filepath)
    return filepath


def register_temp_dir(dirpath: str) -> str:
    """
    Register a temporary directory for cleanup at process exit.

    Parameters
    ----------
    dirpath : str
        The path to the temporary directory.

    Returns
    -------
    str
        The same dirpath (for convenience in chaining).
    """
    _temp_dirs.add(dirpath)
    return dirpath


def _cleanup():
    """Clean up all registered temporary files and directories."""
    for filepath in _temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass
    for dirpath in _temp_dirs:
        try:
            if os.path.exists(dirpath):
                shutil.rmtree(dirpath)
        except OSError:
            pass


atexit.register(_cleanup)
