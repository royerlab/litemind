import numpy as np

from litemind.media.types.media_image import Image
from litemind.ressources.media_resources import MediaResources


class TestImage:
    def test_init_with_path(self):
        """Test initializing Image with a path."""
        image_uri = MediaResources.get_local_test_image_uri("future.jpeg")

        # Instantiate Image:
        image_media = Image(uri=image_uri)

        # Chck URI:
        assert image_media.uri == image_uri

    def test_init_with_array(self):
        """Test initializing Image with an array."""
        array = np.zeros((100, 100, 3), dtype=np.uint8)
        image_media = Image.from_data(array=array)
        assert np.array_equal(image_media.array, array)
        assert image_media.uri is not None

    def test_str(self):
        """Test string representation of Image."""

        # Image URI:
        uri = MediaResources.get_local_test_image_uri("panda.jpg")

        # Create Image media:
        image_media = Image(uri=uri)

        assert "panda.jpg" in str(image_media)
