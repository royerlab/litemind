import base64
import http.server
import os
import socketserver
import tempfile
import threading

import pytest

from litemind.utils.normalise_uri_to_local_file_path import (
    _temp_directories,
    uri_to_local_file_path,
)


@pytest.fixture
def temp_local_file():
    """Create a temporary file with known content."""
    fd, path = tempfile.mkstemp()
    with os.fdopen(fd, "wb") as f:
        f.write(b"some test content")
    yield path
    os.remove(path)


class ContentHolder:
    content = b"test content from server"
    user_agent = None


@pytest.fixture
def ephemeral_http_server():
    """Start an HTTP server on an ephemeral port."""

    class TestHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            ContentHolder.user_agent = self.headers.get("User-Agent")
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(ContentHolder.content)

        def log_message(self, *args, **kwargs):
            return  # Silence logs

    with socketserver.TCPServer(("127.0.0.1", 0), TestHandler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()

        file_url = f"http://127.0.0.1:{port}/file"
        yield file_url, ContentHolder

        httpd.shutdown()
        thread.join()


def test_local_file_absolute(temp_local_file):
    """An absolute local file path should return the same absolute path."""
    result_path = uri_to_local_file_path(temp_local_file)
    assert os.path.abspath(result_path) == os.path.abspath(temp_local_file)
    assert os.path.isfile(result_path)
    with open(result_path, "rb") as f:
        assert f.read() == b"some test content"


def test_local_file_relative(temp_local_file):
    """A relative local file path should be resolved to absolute."""
    # Skip this test if we're in the root directory
    if os.path.dirname(temp_local_file) == "/":
        pytest.skip("Test not applicable in root directory")

    relative_path = os.path.relpath(temp_local_file)
    result_path = uri_to_local_file_path(relative_path)
    assert os.path.abspath(result_path) == os.path.abspath(temp_local_file)
    assert os.path.isfile(result_path)


def test_file_uri(temp_local_file):
    """A file:// URI should return the correct absolute local path."""
    file_uri = f"file://{temp_local_file}"
    result_path = uri_to_local_file_path(file_uri)
    assert os.path.abspath(result_path) == os.path.abspath(temp_local_file)
    assert os.path.isfile(result_path)


def test_file_uri_nonexistent():
    """A file:// URI to a nonexistent file should raise an error."""
    nonexistent_path = "/nonexistent/file/path.txt"
    file_uri = f"file://{nonexistent_path}"
    with pytest.raises(ValueError) as excinfo:
        uri_to_local_file_path(file_uri)
    assert "not found" in str(excinfo.value)


def test_remote_download(ephemeral_http_server):
    """Test downloading a file from a remote HTTP server."""
    url, content_holder = ephemeral_http_server

    result_path = uri_to_local_file_path(url)
    try:
        # Check the downloaded file exists and has correct content
        assert os.path.isfile(result_path)
        with open(result_path, "rb") as f:
            assert f.read() == content_holder.content

        # Verify User-Agent was sent
        assert content_holder.user_agent is not None
        assert len(content_holder.user_agent) > 0
    finally:
        # The cleanup should be handled by the atexit handler
        pass


def test_data_uri_pdf():
    """A data: URI with base64 encoding should be properly decoded."""
    original_content = b"PDFDATA"
    encoded = base64.b64encode(original_content).decode("utf-8")
    data_uri = f"data:application/pdf;base64,{encoded}"

    result_path = uri_to_local_file_path(data_uri)
    try:
        assert os.path.isfile(result_path)
        assert result_path.endswith(
            ".pdf"
        ), "Should use correct extension from MIME type"
        with open(result_path, "rb") as f:
            assert f.read() == original_content
    finally:
        # Cleanup handled by atexit
        pass


def test_data_uri_image():
    """Test data URI with image MIME type."""
    original_content = b"IMAGEDATA"
    encoded = base64.b64encode(original_content).decode("utf-8")
    data_uri = f"data:image/png;base64,{encoded}"

    result_path = uri_to_local_file_path(data_uri)
    try:
        assert os.path.isfile(result_path)
        assert result_path.endswith(
            ".png"
        ), "Should use correct extension from MIME type"
        with open(result_path, "rb") as f:
            assert f.read() == original_content
    finally:
        # Cleanup handled by atexit
        pass


def test_raw_base64():
    """Test raw base64 data."""
    original_content = b"RawBase64Content"
    encoded = base64.b64encode(original_content).decode("utf-8")

    result_path = uri_to_local_file_path(encoded)
    try:
        assert os.path.isfile(result_path)
        with open(result_path, "rb") as f:
            assert f.read() == original_content
    finally:
        # Cleanup handled by atexit
        pass


def test_invalid_uri():
    """Test invalid URI formats."""
    with pytest.raises(ValueError):
        uri_to_local_file_path("")

    with pytest.raises(ValueError):
        uri_to_local_file_path("invalid:scheme://something")


def test_invalid_base64():
    """Test with invalid base64 data."""
    invalid_b64 = "!!!this is not base64!!!"

    with pytest.raises(ValueError) as exc:
        uri_to_local_file_path(invalid_b64)

    # Check for appropriate error message
    error_msg = str(exc.value).lower()
    assert "base64" in error_msg


def test_nonexistent_remote_url():
    """Test with a URL that doesn't exist."""
    with pytest.raises(ValueError) as exc:
        uri_to_local_file_path("https://nonexistent.example.com/file.txt")
    assert "failed to download" in str(exc.value).lower()


def test_invalid_data_uri():
    """Test malformed data URI."""
    with pytest.raises(ValueError):
        uri_to_local_file_path("data:application/pdf;base64")  # Missing comma and data


def test_temp_directory_tracking():
    """Test that temporary directories are properly tracked for cleanup."""
    # Record the number of temp directories before
    num_dirs_before = len(_temp_directories)

    # Create a data URI that will generate a temp directory
    data = base64.b64encode(b"test").decode("utf-8")
    data_uri = f"data:text/plain;base64,{data}"

    # Process it
    result_path = uri_to_local_file_path(data_uri)
    temp_dir = os.path.dirname(result_path)

    # Verify the directory exists and is tracked
    assert os.path.exists(temp_dir)
    assert temp_dir in _temp_directories
    assert len(_temp_directories) > num_dirs_before


def test_padding_correction():
    """Test that base64 with missing padding is corrected."""
    # Base64 that needs padding (should end with ==)
    original = b"needs padding"
    encoded = base64.b64encode(original).decode("utf-8").rstrip("=")

    result_path = uri_to_local_file_path(encoded)
    try:
        with open(result_path, "rb") as f:
            assert f.read() == original
    finally:
        # Cleanup handled by atexit
        pass
