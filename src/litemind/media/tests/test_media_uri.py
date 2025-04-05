import tempfile
from pathlib import Path

import pytest

from litemind.agent.messages.message_block import MessageBlock
from litemind.media.media_uri import MediaURI


class MinimalMediaURI(MediaURI):
    """Minimal concrete implementation of MediaURI for testing"""

    def load_from_uri(self):
        # Simple implementation of the abstract method
        pass


class TestMediaURI:
    @pytest.fixture
    def setup_test_files(self):
        """Create temporary test files for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test text file
            text_file = Path(temp_dir) / "test.txt"
            with open(text_file, "w") as f:
                f.write("test content")

            # Create test image file
            image_file = Path(temp_dir) / "test.jpg"
            with open(image_file, "wb") as f:
                f.write(b"dummy image data")

            yield {
                "text_file": str(text_file),
                "image_file": str(image_file),
                "temp_dir": temp_dir,
            }

    def test_init_with_uri(self):
        """Test initialization with different URIs"""
        # Test with remote URI
        remote_uri = "https://example.com/image.jpg"
        media = MinimalMediaURI(uri=remote_uri)
        assert media.uri == remote_uri
        assert media.extension is None

        # Test with local URI
        local_uri = "file:///path/to/file.txt"
        media = MinimalMediaURI(uri=local_uri)
        assert media.uri == local_uri

        # Test with explicit extension
        media = MinimalMediaURI(uri=remote_uri, extension="png")
        assert media.extension == "png"

    def test_is_local(self):
        """Test is_local method"""
        # Test with remote URI
        remote_uri = "https://example.com/image.jpg"
        media = MinimalMediaURI(uri=remote_uri)
        assert not media.is_local()

        # Test with file:// URI
        file_uri = "file:///path/to/file.txt"
        media = MinimalMediaURI(uri=file_uri)
        assert media.is_local()

        # Test with absolute path
        abs_path_uri = "file:///path/to/file.txt"
        media = MinimalMediaURI(uri=abs_path_uri)
        assert media.is_local()

    def test_get_filename(self):
        """Test get_filename method"""
        # Simple case
        media = MinimalMediaURI(uri="https://example.com/image.jpg")
        assert media.get_filename() == "image.jpg"

        # Path with multiple slashes
        media = MinimalMediaURI(uri="https://example.com/path/to/image.jpg")
        assert media.get_filename() == "image.jpg"

        # Local file URI
        media = MinimalMediaURI(uri="file:///path/to/document.pdf")
        assert media.get_filename() == "document.pdf"

        # Empty URI
        try:
            media = MinimalMediaURI(uri="")
            assert False
        except ValueError:
            assert True

    def test_get_extension(self):
        """Test get_extension method"""
        # From URI
        media = MinimalMediaURI(uri="https://example.com/image.jpg")
        assert media.get_extension() == "jpg"

        # Explicit extension overrides URI
        media = MinimalMediaURI(uri="https://example.com/image.jpg", extension="png")
        assert media.get_extension() == "png"

        # URI without extension but with explicit extension
        media = MinimalMediaURI(uri="https://example.com/image", extension="jpg")
        assert media.get_extension() == "jpg"

    def test_has_extension(self):
        """Test has_extension method"""
        # Simple case
        media = MinimalMediaURI(uri="https://example.com/image.jpg")
        assert media.has_extension("jpg")
        assert not media.has_extension("png")

        # Case insensitive match
        media = MinimalMediaURI(uri="https://example.com/image.JPG")
        assert media.has_extension("jpg")

        # With explicit extension
        media = MinimalMediaURI(uri="https://example.com/image", extension="jpg")
        assert media.has_extension("jpg")

    def test_to_remote_or_data_uri(self, setup_test_files, monkeypatch):
        """Test to_remote_or_data_uri method"""
        # For remote URI, should return as is
        remote_uri = "https://example.com/image.jpg"
        media = MinimalMediaURI(uri=remote_uri)
        assert media.to_remote_or_data_uri() == remote_uri

        # For local file URI, we need to mock the dependencies since this is a unit test
        file_path = setup_test_files["text_file"]
        file_uri = f"file://{file_path}"

        # Create a real test with the actual file
        media = MinimalMediaURI(uri=file_uri)
        data_uri = media.to_remote_or_data_uri()

        # Verify it's a data URI
        assert data_uri.startswith("data:")
        assert ";base64," in data_uri

    def test_to_message_block(self):
        """Test to_message_block method"""
        media = MinimalMediaURI(uri="https://example.com/image.jpg")
        message_block = media.to_message_block()

        assert isinstance(message_block, MessageBlock)
        assert message_block.media == media

    def test_string_representation(self):
        """Test string representation methods"""
        uri = "https://example.com/image.jpg"
        media = MinimalMediaURI(uri=uri)

        # Test __str__
        assert str(media) == uri

        # Test __repr__
        assert repr(media) == f"MediaURI({uri})"

        # Test __len__
        assert len(media) == len(uri)
