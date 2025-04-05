import os
import tempfile

import pytest
from PIL import Image

from litemind.utils.convert_image_to_jpg import convert_image_to_jpeg


@pytest.fixture
def create_temp_image():
    temp_image_path = os.path.join(tempfile.gettempdir(), "test_image.png")
    image = Image.new("RGB", (100, 100), color="red")
    image.save(temp_image_path)
    yield temp_image_path
    os.remove(temp_image_path)


def test_convert_image_to_jpeg(create_temp_image):
    jpeg_path = convert_image_to_jpeg(create_temp_image)
    assert os.path.exists(jpeg_path)
    assert jpeg_path.endswith(".jpg")
    with Image.open(jpeg_path) as img:
        assert img.format == "JPEG"
    os.remove(jpeg_path)
