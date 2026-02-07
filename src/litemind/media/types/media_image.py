from typing import Optional

import numpy

from litemind.media.media_uri import MediaURI
from litemind.utils.convert_image_to_jpg import convert_image_to_jpeg
from litemind.utils.convert_image_to_png import convert_image_to_png
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class Image(MediaURI):
    """Media that stores an image referenced by URI.

    Supports loading from local files or remote URLs. Images can be
    converted between formats (PNG, JPEG) and loaded as numpy arrays
    or PIL Image objects.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """Create a new image media.

        Parameters
        ----------
        uri : str
            The image URI or local file path.
        extension : str, optional
            File extension override without the leading dot (e.g. ``"png"``).
        **kwargs
            Additional keyword arguments forwarded to ``MediaURI``.
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
        """Create an Image from a numpy array.

        The array is converted to a PIL Image, saved to disk (as PNG by
        default), and wrapped in an Image media.

        Parameters
        ----------
        array : numpy.ndarray
            Pixel data as a numpy array (H x W or H x W x C).
        filepath : str, optional
            Destination file path. A temporary file is created if None.
        format : str, optional
            Image format for saving (e.g. ``"PNG"``, ``"JPEG"``).

        Returns
        -------
        Image
            A new Image media referencing the saved file.
        """

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
        """Create an Image from a PIL Image object.

        Parameters
        ----------
        image : PIL.Image.Image
            The PIL image to save and wrap.
        filepath : str, optional
            Destination file path. A temporary PNG file is created if None.
        format : str, optional
            Image format for saving (e.g. ``"PNG"``, ``"JPEG"``).

        Returns
        -------
        Image
            A new Image media referencing the saved file.
        """

        if filepath is None:
            # Create temporary file:
            import tempfile

            from litemind.utils.temp_file_manager import register_temp_file

            filepath = register_temp_file(
                tempfile.NamedTemporaryFile(delete=False).name + ".png"
            )

        # Write
        image.save(filepath, format)

        # URI:
        image_uri = "file://" + filepath

        # Create Image from URI  and array:
        image = Image(uri=image_uri)

        return image

    def load_from_uri(self):
        """Load the image data from the URI into ``self.array``.

        Downloads the image if necessary, opens it with PIL, and converts
        it to a numpy array.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist at the resolved path.
        Exception
            If the image cannot be opened or decoded.
        """

        # Download the image file from the URI to a local file:
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
        """Open the image as a PIL Image object.

        Parameters
        ----------
        normalise_to_png : bool, optional
            If True, convert the image to PNG format before opening.
            Default is False.

        Returns
        -------
        PIL.Image.Image
            The opened PIL image.
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
        """Convert the image to PNG format in place.

        Updates ``self.local_path`` to point to the converted file.

        Returns
        -------
        str
            The local file path of the PNG image.
        """

        # Ensure local file is available:
        self.to_local_file_path()

        # Ensure local file is a PNG file:
        self.local_path = convert_image_to_png(self.local_path)

        return self.local_path

    def normalise_to_jpeg(self):
        """Convert the image to JPEG format in place.

        Updates ``self.local_path`` to point to the converted file.

        Returns
        -------
        str
            The local file path of the JPEG image.
        """

        # Ensure local file is available:
        self.to_local_file_path()

        # Ensure local file is a JPEG file:
        self.local_path = convert_image_to_jpeg(self.local_path)

        return self.local_path
