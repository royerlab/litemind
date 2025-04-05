import os
import tempfile

import pytest

from litemind.utils.read_file_and_convert_to_base64 import (
    read_file_and_convert_to_base64,
)


@pytest.fixture
def create_temp_file():
    temp_file_path = os.path.join(tempfile.gettempdir(), "test_file.txt")
    with open(temp_file_path, "w") as file:
        file.write("This is a tests file.")
    yield temp_file_path
    os.remove(temp_file_path)


def test_read_file_and_convert_to_base64(create_temp_file):
    base64_string = read_file_and_convert_to_base64(create_temp_file)
    assert base64_string == "VGhpcyBpcyBhIHRlc3RzIGZpbGUu"
