import tempfile

from PIL import Image


def convert_image_to_jpeg(file_path: str) -> str:
    """
    Convert an image to JPEG format.

    Opens the image with PIL and saves it as a JPEG temporary file.
    If the file is already a JPEG, returns the original path unchanged.

    Parameters
    ----------
    file_path : str
        The path to the image file to convert.

    Returns
    -------
    str
        The path to the converted JPEG file, or the original path if
        already JPEG.

    Raises
    ------
    ValueError
        If the file path is empty or the file is not a valid image.
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
                # Register for cleanup at exit:
                from litemind.utils.temp_file_manager import register_temp_file

                register_temp_file(temp_file.name)

                # Save the image as a JPEG file:
                img.save(temp_file.name, "JPEG")

                return temp_file.name
    except IOError:
        raise ValueError(f"File '{file_path}' is not a valid image file.")
