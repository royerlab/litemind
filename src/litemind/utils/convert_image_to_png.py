"""Utilities for converting images to PNG format."""

import os
import tempfile

from PIL import Image


def convert_image_to_png(file_path: str) -> str:
    """
    Convert an image to PNG format.

    Opens the image with PIL and saves it as a PNG temporary file.
    If the file is already a PNG, returns the original path unchanged.

    Parameters
    ----------
    file_path : str
        The path to the image file to convert.

    Returns
    -------
    str
        The path to the converted PNG file, or the original path if
        already PNG.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is not a valid image.
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
                # Register for cleanup at exit:
                from litemind.utils.temp_file_manager import register_temp_file

                register_temp_file(temp_file.name)

                # Save the image as a PNG file:
                img.save(temp_file.name, "PNG")

                # Return the path to the PNG file:
                return temp_file.name
    except IOError:
        raise ValueError(f"The file {file_path} is not a valid image.")
