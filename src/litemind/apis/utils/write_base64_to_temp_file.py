import base64
import re
import tempfile


def write_base64_to_temp_file(data_uri: str) -> str:
    """
    Saves a data URI (data:image/...;base64,...) to a temporary file.
    Returns the local file path.
    """

    match = re.match(r"data:image/(png|jpeg|jpg);base64,(.*)", data_uri,
                     re.IGNORECASE)
    if not match:
        raise ValueError("Invalid data URI format")

    image_type = match.group(1).lower()
    extension = ".png" if image_type == "png" else ".jpg"
    base64_part = match.group(2)

    image_data = base64.b64decode(base64_part)

    with tempfile.NamedTemporaryFile(suffix=extension,
                                     delete=False) as tmp_file:
        tmp_file.write(image_data)
        tmp_path = tmp_file.name

    return tmp_path
