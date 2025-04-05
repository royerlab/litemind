import os
import tempfile

from PIL import Image


def convert_image_to_png(file_path: str) -> str:
    """
    Convert an image to a PNG file.
    Uses PIL to open the image in any format and save it as a PNG file.

    Parameters
    ----------
    file_path : str
        The path to the image file.

    Returns
    -------
    str
        The path to the PNG file.

    """

    # Check if the file exists:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Check if the file is already a PNG:
    if file_path.lower().endswith(".png"):
        return file_path

    # Check if the file is a valid image:
    try:
        # Open the image file:
        with Image.open(file_path) as img:
            # Create a temporary file to save the image:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                # Save the image as a PNG file:
                img.save(temp_file.name, "PNG")

                # Return the path to the PNG file:
                return temp_file.name
    except IOError:
        raise ValueError(f"The file {file_path} is not a valid image.")
