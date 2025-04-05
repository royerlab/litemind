import os.path
from urllib.parse import urlparse


def is_uri(uri_str: str) -> bool:
    """
    Check if the given string is a valid URI.

    Parameters
    ----------
    uri_str : str
        The string to check.

    Returns
    -------
    bool
        True if the string is a valid URI, False otherwise.
    """
    try:
        result = urlparse(uri_str)

        # Validate the path component
        if result.path and not is_valid_uri_path(result.path):
            return False

        # Validate components
        if result.scheme and not is_valid_uri_scheme(result.scheme):
            return False

        if result.netloc and not is_valid_uri_netloc(result.netloc):
            return False

        # Check for empty scheme or double slashes at start of path
        if result.path and result.path.startswith("//") and not result.netloc:
            return False

        return all([result.scheme, result.netloc]) or all([result.scheme, result.path])
    except ValueError:
        return False


def is_valid_uri_scheme(scheme: str) -> bool:
    """
    Validate a URI scheme according to RFC 3986.

    Parameters
    ----------
    scheme : str
        The URI scheme to validate

    Returns
    -------
    bool
        True if the scheme is valid, False otherwise
    """
    if not scheme:
        return False

    # Scheme must start with a letter and contain only letters, digits, +, ., or -
    if not scheme[0].isalpha():
        return False

    return all(c.isalnum() or c in "+-." for c in scheme)


def is_valid_uri_netloc(netloc: str) -> bool:
    """
    Validate a URI netloc according to RFC 3986.

    Parameters
    ----------
    netloc : str
        The URI netloc to validate

    Returns
    -------
    bool
        True if the netloc is valid, False otherwise
    """
    # Empty netloc can be valid for some URIs (like file:///path)
    if not netloc:
        return True

    # Check for illegal characters
    if "\\" in netloc or " " in netloc:
        return False

    # Simple check for basic structure
    # A more comprehensive check would validate the hostname format
    if "@" in netloc:
        userinfo, hostname = netloc.rsplit("@", 1)
        if not userinfo:
            return False
    else:
        hostname = netloc

    if ":" in hostname:
        host, port = hostname.rsplit(":", 1)
        # Validate that port is numeric
        if not port.isdigit():
            return False

    return True


def is_valid_uri_path(path_str: str) -> bool:
    """
    Validate a URI path component according to RFC 3986.

    Parameters
    ----------
    path_str : str
        The URI path string to validate

    Returns
    -------
    bool
        True if the path is valid, False otherwise
    """
    # Empty paths are allowed in some URIs
    if path_str is None:
        return True

    # Check for backslashes which are not allowed in URI paths
    if "\\" in path_str:
        return False

    # Check for proper percent-encoding
    i = 0
    while i < len(path_str):
        if path_str[i] == "%":
            # Each percent encoding must be followed by two hex digits
            if i + 2 >= len(path_str):
                return False
            try:
                int(path_str[i + 1 : i + 3], 16)
                i += 3
            except ValueError:
                return False
        else:
            i += 1

    return True


def is_valid_path(path):
    """
    Check if a file path is syntactically valid.
    This doesn't check if the path exists, only if it's well-formed.

    Args:
        path: The file path to check

    Returns:
        bool: True if the path is syntactically valid, False otherwise
    """
    # Check if the path is a string
    if not isinstance(path, str):
        return False

    # Use os.path.normpath to normalize the path
    # If the result is different from the original in a way that
    # indicates invalid syntax, the path is likely invalid
    normalized = os.path.normpath(path)

    # On Windows, check for invalid characters
    if os.name == "nt":
        # Windows has specific invalid characters
        invalid_chars = ["<", ">", ":", '"', "|", "?", "*"]
        if any(char in normalized for char in invalid_chars):
            return False

    # Check for other common indicators of invalid paths
    if os.path.splitdrive(normalized)[1].startswith("//"):
        return False

    return True
