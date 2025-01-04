import base64


def read_file_and_convert_to_base64(file_path: str) -> str:
    """
    Get the Base64 encoding of the file at the given path.

    Parameters
    ----------
    file_path : str
        The path to the file to encode.

    Returns
    -------
    str
        The Base64 encoding of the file.

    """
    with open(file_path, "rb") as file:
        file_data = file.read()
        base64_data = base64.b64encode(file_data)
        base64_string = base64_data.decode("utf-8")
    return base64_string


def base64_to_data_uri(base64_string: str, media_type: str) -> str:
    """
    Convert a Base64 string to a data URI.

    Parameters
    ----------
    base64_string : str
        The Base64 string to convert.
    media_type : str
        The media type of the data.

    Returns
    -------
    str
        The data URI.

    """
    return f"data:{media_type};base64,{base64_string}"
