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

    # Open the image file:
    with Image.open(file_path) as img:
        # Create a temporary file to save the image:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            # Save the image as a JPEG file:
            img.save(temp_file.name, "JPEG")

            return temp_file.name
