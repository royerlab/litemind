import os

from litemind.utils.dowload_image_to_tempfile import \
    download_image_to_temp_file


def test_download_image_to_temp_file():
    # Use the specified image URL
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/Example.png'

    # Call the function
    temp_file_path = download_image_to_temp_file(url)

    # Check that the temp file was created
    assert os.path.exists(temp_file_path)

    # Check that the temp file contains some content
    with open(temp_file_path, 'rb') as f:
        content = f.read()
        assert len(content) > 0

    # Clean up the temp file
    os.remove(temp_file_path)
