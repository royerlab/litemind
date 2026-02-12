"""Utilities for reading files and converting them to base64-encoded strings."""

import base64


def read_file_and_convert_to_base64(file_path: str) -> str:
    """
    Read a file and return its contents as a base64-encoded string.

    Parameters
    ----------
    file_path : str
        The path to the file to encode.

    Returns
    -------
    str
        The base64-encoded contents of the file.
    """
    with open(file_path, "rb") as file:
        file_data = file.read()
        base64_data = base64.b64encode(file_data)
        base64_string = base64_data.decode("utf-8")
    return base64_string


def base64_to_data_uri(base64_string: str, media_type: str) -> str:
    """
    Convert a base64 string to a data URI.

    Parameters
    ----------
    base64_string : str
        The base64-encoded data.
    media_type : str
        The MIME type of the data (e.g., ``"image/png"``).

    Returns
    -------
    str
        The data URI in the format ``data:<media_type>;base64,<data>``.
    """
    return f"data:{media_type};base64,{base64_string}"
