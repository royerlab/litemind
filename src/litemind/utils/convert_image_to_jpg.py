import tempfile

from PIL import Image


def convert_image_to_jpeg(file_path: str) -> str:
    """
    Convert an image to a JPEG file.
    Uses PIL to open the image in any format and save it as a JPEG file.

    Parameters
    ----------
    file_path : str
        The path to the image file.

    Returns
    -------
    str
        The path to the JPEG file.

    """

    # Check if the file exists:
    if not file_path:
        raise ValueError("File path cannot be empty.")

    # Check if the file is already a JPEG:
    if file_path.lower().endswith(".jpg") or file_path.lower().endswith(".jpeg"):
        return file_path

    # Check if the file is a valid image:
    try:
        # Open the image file:
        with Image.open(file_path) as img:
            # Create a temporary file to save the image:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
                # Save the image as a JPEG file:
                img.save(temp_file.name, "JPEG")

                return temp_file.name
    except IOError:
        raise ValueError(f"File '{file_path}' is not a valid image file.")
