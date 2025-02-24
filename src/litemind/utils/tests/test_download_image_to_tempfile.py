import os

from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


def test_download_image_to_temp_file():
    # Use the specified image URL
    url = "https://upload.wikimedia.org/wikipedia/commons/7/70/Example.png"

    # Call the function
    temp_file_path = uri_to_local_file_path(url)

    # Check that the temp file was created
    assert os.path.exists(temp_file_path)

    # Check that the temp file contains some content
    with open(temp_file_path, "rb") as f:
        content = f.read()
        assert len(content) > 0

    # Clean up the temp file
    os.remove(temp_file_path)
