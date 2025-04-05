import os
import tempfile

import pytest
from PIL import Image

from litemind.utils.convert_image_to_png import convert_image_to_png


@pytest.fixture
def create_temp_image():
    temp_image_path = os.path.join(tempfile.gettempdir(), "test_image.jpg")
    image = Image.new("RGB", (100, 100), color="red")
    image.save(temp_image_path)
    yield temp_image_path
    os.remove(temp_image_path)


def test_convert_image_to_png(create_temp_image):
    png_path = convert_image_to_png(create_temp_image)
    assert os.path.exists(png_path)
    assert png_path.endswith(".png")
    with Image.open(png_path) as img:
        assert img.format == "PNG"
    os.remove(png_path)
