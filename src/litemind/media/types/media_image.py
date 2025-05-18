from typing import Optional

import numpy

from litemind.media.media_uri import MediaURI
from litemind.utils.convert_image_to_jpg import convert_image_to_jpeg
from litemind.utils.convert_image_to_png import convert_image_to_png
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class Image(MediaURI):
    """
    A media that stores an image
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """
        Create a new image media.

        Parameters
        ----------
        uri: str
            The image URI.
        extension: str
            Extension/Type of the image file in case it is not clear from the URI. This is the extension _without_ the dot -- 'png' not '.png'.
        kwargs: dict
            Other arguments passed to MediaURI.

        """

        super().__init__(uri=uri, extension=extension, **kwargs)

        # Set attributes:
        self.array = None

    @classmethod
    def from_data(
        cls,
        array: numpy.ndarray,
        filepath: Optional[str] = None,
        format: Optional[str] = None,
    ):

        # Create image using PIL
        from PIL import Image as PILImage

        # Load image:
        image = PILImage.fromarray(array)

        # Save image as PNG:
        image_media = Image.from_PIL_image(
            image=image, filepath=filepath, format=format
        )

        image_media.array = array

        return image_media

    @classmethod
    def from_PIL_image(
        cls,
        image: "PILImage",
        filepath: Optional[str] = None,
        format: Optional[str] = None,
    ):
        # Create image using PIL

        # Save image as PNG:

        if filepath is None:
            # Create temporary file:
            import tempfile

            filepath = tempfile.NamedTemporaryFile(delete=False).name + ".png"

        # Write
        image.save(filepath, format)

        # URI:
        image_uri = "file://" + filepath

        # Create Image from URI  and array:
        image = Image(uri=image_uri)

        return image

    def load_from_uri(self):

        # Download the video file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # load the image using PIL:
        from PIL import Image as PILImage

        try:
            # Open the image file:
            image_file = PILImage.open(local_file)

            # Convert to numpy array:
            self.array = numpy.array(image_file)

            # Close the image file:
            image_file.close()
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at URI: {self.uri}")
        except Exception as e:  # Catching PIL's specific exception is a bit tricky
            raise Exception(
                f"Could not open or decode image file: {self.uri}. Error: {e}"
            )

    def open_pil_image(self, normalise_to_png: bool = False) -> "ImageFile":
        """
        Convert the image to a PIL image.

        Parameters
        ----------
        normalise_to_png: bool
            If True, convert the image to PNG format.

        Returns
        -------
        PILImage
            The image as a PIL image.
        """

        # load the image using PIL:
        from PIL import Image as PILImage

        # Convert the image URI to a local file path:
        local_path = uri_to_local_file_path(self.uri)

        if normalise_to_png:
            local_path = convert_image_to_png(local_path)

        # Open the image file and return the ImageFile object:
        return PILImage.open(local_path)

    def normalise_to_png(self):
        """
        Convert the image to PNG format.
        This method ensures that the image is in PNG format by converting it if necessary.

        Returns
        -------
        str
            The local path of the image in PNG format.

        """

        # Ensure local file is available:
        self.to_local_file_path()

        # Ensure local file is a PNG file:
        self.local_path = convert_image_to_png(self.local_path)

        return self.local_path

    def normalise_to_jpeg(self):
        """
        Convert the image to JPEG format.
        This method ensures that the image is in JPEG format by converting it if necessary.

        Returns
        -------
        str
            The local path of the image in JPEG format.

        """

        # Ensure local file is available:
        self.to_local_file_path()

        # Ensure local file is a PNG file:
        self.local_path = convert_image_to_jpeg(self.local_path)

        return self.local_path
