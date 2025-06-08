import binascii
import os
import tempfile

from litemind.media.types.media_file import File
from litemind.media.types.media_text import Text


class TestFile:
    def test_initialization(self):
        """Test that the File class can be initialized with a valid URI."""
        # Create a temporary file since we need an actual file
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp:
            temp.write(b"test content")
            temp.flush()
            uri = f"file://{temp.name}"
            file_media = File(uri=uri)
            assert file_media.uri == uri

    def test_to_text_media_with_test_file(self):
        """Test to_text_media method with test file from resources."""
        # Create a temporary file to use for testing
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as temp:
            temp.write(b"Test binary data for file media")
            temp_path = temp.name

        try:
            uri = f"file://{temp_path}"
            file_media = File(uri=uri)

            # Get text media representation
            text_media = file_media.to_markdown_text_media()

            # Assertions
            assert isinstance(text_media, Text)

            text_content = text_media.text

            # Check content structure
            assert "Description of file:" in text_content
            assert f"- Absolute Path: {os.path.abspath(temp_path)}" in text_content
            assert "- File Size:" in text_content

            # Test file should have a mime type
            assert "- MIME Type:" in text_content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_to_text_media_small_file(self):
        """Test to_text_media method with a small file."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as temp:
            # Write 50 bytes of data
            content = bytes([i % 256 for i in range(50)])
            temp.write(content)
            temp_path = temp.name

        try:
            uri = f"file://{temp_path}"
            file_media = File(uri=uri)

            # Get text media representation
            text_media = file_media.to_markdown_text_media(hex_dump_length=128)

            # Assertions
            assert isinstance(text_media, Text)

            text_content = text_media.text

            # Check content structure
            assert "Description of file:" in text_content
            assert f"- Absolute Path: {os.path.abspath(temp_path)}" in text_content
            assert "- File Size: 50 bytes" in text_content
            assert "Entire file content (50 bytes):" in text_content

            # Verify the hex content
            expected_hex = binascii.hexlify(content).decode("ascii")
            # Check for hex content (without spaces)
            assert expected_hex[:10] in text_content.replace(" ", "")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_to_text_media_large_file(self):
        """Test to_text_media method with a larger file."""
        hex_dump_length = 64  # Smaller value for testing

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as temp:
            # Write 300 bytes of data
            content = bytes([i % 256 for i in range(300)])
            temp.write(content)
            temp_path = temp.name

        try:
            uri = f"file://{temp_path}"
            file_media = File(uri=uri)

            # Get text media representation
            text_media = file_media.to_markdown_text_media(
                hex_dump_length=hex_dump_length
            )

            # Assertions
            assert isinstance(text_media, Text)

            text_content = text_media.text

            # Check content structure
            assert "Description of file:" in text_content
            assert f"- Absolute Path: {os.path.abspath(temp_path)}" in text_content
            assert "- File Size: 300 bytes" in text_content
            assert f"First {hex_dump_length} bytes (hex):" in text_content
            assert f"Last {hex_dump_length} bytes (hex):" in text_content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_to_text_media_empty_file(self):
        """Test to_text_media method with an empty file."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as temp:
            temp_path = temp.name

        try:
            uri = f"file://{temp_path}"
            file_media = File(uri=uri)

            # Get text media representation
            text_media = file_media.to_markdown_text_media()

            # Assertions
            assert isinstance(text_media, Text)

            text_content = text_media.text

            # Check content
            assert "Description of file:" in text_content
            assert f"- Absolute Path: {os.path.abspath(temp_path)}" in text_content
            assert "- File Size: 0 bytes" in text_content
            assert "File is empty." in text_content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_to_text_media_binary_content(self):
        """Test to_text_media method with mixed binary content."""
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as temp:
            # Include null bytes and other non-printable characters
            content = bytes(
                [0, 255, 10, 13, 27, 7, 9] + [65 + i % 26 for i in range(20)]
            )
            temp.write(content)
            temp_path = temp.name

        try:
            uri = f"file://{temp_path}"
            file_media = File(uri=uri)

            # Get text media representation
            text_media = file_media.to_markdown_text_media()

            # Assertions
            text_content = text_media.text

            # Verify file size is correct
            assert f"- File Size: {len(content)} bytes" in text_content

            # Check that hex representation includes our specific byte values
            hex_content = text_content.replace(" ", "").lower()
            assert "00" in hex_content  # null byte
            assert "ff" in hex_content  # 255 byte
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_to_text_media_error_handling(self):
        """Test error handling with non-existent file."""
        non_existent_path = "/path/to/nonexistent/file.bin"
        uri = f"file://{non_existent_path}"
        file_media = File(uri=uri)

        try:
            # Get text media representation
            text_media = file_media.to_markdown_text_media()

            # Should not reach this point!
            assert False, "Expected an error but none was raised."

        except ValueError:
            # Expected ValueError due to non-existent file
            assert True, "Expected an error but none was raised."

    def test_load_from_uri(self):
        """Test the load_from_uri method."""
        with tempfile.NamedTemporaryFile(suffix=".txt") as temp:
            test_content = b"test file content"
            temp.write(test_content)
            temp.flush()

            uri = f"file://{temp.name}"
            file_media = File(uri=uri)

            # Test the load_from_uri method
            loaded_content = file_media.load_from_uri()
            assert loaded_content == test_content
