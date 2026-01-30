from typing import List, Optional, Tuple


def default_folder_scanning_parameters(
    allowed_extensions: Optional[List[str]],
    excluded_files: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """
    Get default folder scanning parameters with fallback values.

    Parameters
    ----------
    allowed_extensions : Optional[List[str]]
        List of allowed file extensions. If None, defaults are used.
    excluded_files : Optional[List[str]]
        List of files to exclude. If None, defaults are used.

    Returns
    -------
    Tuple[List[str], List[str]]
        A tuple of (allowed_extensions, excluded_files) with defaults applied.
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
