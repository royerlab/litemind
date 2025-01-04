import os

from src.litemind.apis.utils.write_base64_to_temp_file import \
    write_base64_to_temp_file


def test_save_base64_to_temp_file():
    # Example base64 data URI
    data_uri = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA' \
               'AAAFCAYAAACNbyblAAAAHElEQVQI12P4' \
               '//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=='

    # Call the function
    temp_file_path = write_base64_to_temp_file(data_uri)

    # Check that the temp file was created
    assert os.path.exists(temp_file_path)

    # Check that the temp file contains some content
    with open(temp_file_path, 'rb') as f:
        content = f.read()
        assert len(content) > 0

    # Clean up the temp file
    os.remove(temp_file_path)
