# tests/test_get_local_file_path.py
import base64
import http.server
import os
import socketserver
import tempfile
import threading

import pytest

from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


@pytest.fixture
def temp_local_file():
    """
    Creates a temporary local file and returns its path.
    Cleans up after the tests.
    """
    with tempfile.NamedTemporaryFile(delete=False) as tf:
        tf.write(b"some tests content")
        tf.flush()
        temp_path = tf.name
    yield temp_path
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def ephemeral_http_server():
    """
    Spins up a simple HTTP server on an ephemeral port to serve a fixed file.

    Yields a (url, content_holder) tuple where:
      - url is the file URL (e.g. http://127.0.0.1:PORT/file)
      - content_holder is a dict holding 'content' and 'user_agent' to
        check what was requested and with which User-Agent.
    """

    class ContentHolder:
        content = b"Hello from tests server!"
        user_agent = None  # We'll store the user agent from the request

    class TestHandler(http.server.SimpleHTTPRequestHandler):
        # We override do_GET to serve the content and capture the User-Agent.
        def do_GET(self):
            # Capture User-Agent
            ContentHolder.user_agent = self.headers.get("User-Agent")

            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(ContentHolder.content)

        def log_message(self, _, *args):
            # Silence the default HTTP server logs
            return

    # Create the server
    with socketserver.TCPServer(("127.0.0.1", 0), TestHandler) as httpd:
        # Get ephemeral port
        port = httpd.server_address[1]
        # Start a thread to handle requests
        thread = threading.Thread(target=httpd.serve_forever)
        thread.daemon = True
        thread.start()

        # Construct a URL pointing to a "file" endpoint
        file_url = f"http://127.0.0.1:{port}/file"

        yield file_url, ContentHolder

        # Shutdown the server
        httpd.shutdown()
        thread.join()


def test_local_file_absolute(temp_local_file):
    """
    An absolute local file path should return the same absolute path.
    """
    result_path = uri_to_local_file_path(temp_local_file)
    assert result_path == os.path.abspath(temp_local_file)
    assert os.path.isfile(result_path)
    with open(result_path, "rb") as f:
        assert f.read() == b"some tests content"


def test_local_file_relative(temp_local_file):
    """
    A relative local file path should be resolved to absolute.
    """
    relative_path = os.path.relpath(temp_local_file)
    result_path = uri_to_local_file_path(relative_path)
    assert result_path == os.path.abspath(relative_path)
    assert os.path.isfile(result_path)


def test_file_uri(temp_local_file):
    """
    A file:// URI should return the correct absolute local path.
    """
    file_uri = f"file://{temp_local_file}"
    result_path = uri_to_local_file_path(file_uri)
    assert result_path == os.path.abspath(temp_local_file)
    assert os.path.isfile(result_path)


def test_remote_download(ephemeral_http_server):
    """
    If the URI is remote (http/https/ftp), the file should be downloaded
    into a temp file, and we capture the User-Agent from the request.
    """
    url, content_holder = ephemeral_http_server

    result_path = uri_to_local_file_path(url)
    # The downloaded file should now exist locally
    assert os.path.isfile(result_path)

    with open(result_path, "rb") as f:
        assert f.read() == content_holder.content

    # Also verify that the server saw a User-Agent from the function's random set
    assert content_holder.user_agent is not None
    # Not specifying the entire set here, but you could do:
    #   assert content_holder.user_agent in expected_user_agents
    # if you want to tests it's one of your 10. For now, just check it's not empty:
    assert len(content_holder.user_agent) > 0


def test_base64_data():
    """
    A raw Base64-encoded string should be decoded to a local file.
    """
    original_content = b"Hello, Base64!"
    encoded = base64.b64encode(original_content).decode("utf-8")

    result_path = uri_to_local_file_path(encoded)
    assert os.path.isfile(result_path)
    with open(result_path, "rb") as f:
        assert f.read() == original_content


def test_data_uri():
    """
    A data: URI with base64 encoding should be properly decoded.
    """
    original_content = b"PDFDATA"
    encoded = base64.b64encode(original_content).decode("utf-8")
    data_uri = f"data:application/pdf;base64,{encoded}"

    result_path = uri_to_local_file_path(data_uri)
    assert os.path.isfile(result_path)
    with open(result_path, "rb") as f:
        assert f.read() == original_content


def test_nonexistent_local_path():
    """
    If a path doesn't exist locally but is valid Base64, we interpret it as Base64.
    """
    valid_b64 = base64.b64encode(b"abc123").decode("utf-8")  # "YWJjMTIz"
    # This string does not exist as a file, but is valid Base64.
    assert not os.path.exists(valid_b64)

    result_path = uri_to_local_file_path(valid_b64)
    assert os.path.isfile(result_path)
    with open(result_path, "rb") as f:
        assert f.read() == b"abc123"


def test_invalid_base64():
    """
    A string that is not a file, not a file:// URI, not a valid http/https/ftp,
    and not valid Base64, should raise an error.
    """
    invalid_b64 = "!!!this is not base64!!!"

    with pytest.raises(Exception) as exc:
        uri_to_local_file_path(invalid_b64)
    # Typically, base64.binascii.Error or ValueError for bad padding, etc.
    assert "decode" in str(exc.value).lower() or "padding" in str(exc.value).lower()


def test_wikimedia_download():
    """
    Download a real image from Wikimedia and check that the local file
    has the expected name and is non-empty.
    """
    url = "https://upload.wikimedia.org/wikipedia/commons/8/84/Alexander_the_Great_mosaic_%28cropped%29.jpg"
    local_file = uri_to_local_file_path(url)

    # 1. Verify the file exists
    assert os.path.isfile(local_file), "Expected a downloaded local file."

    # 2. Check that the filename ends with what we expect from the URL path
    expected_extension = ".jpg"
    assert local_file.endswith(
        expected_extension
    ), f"Downloaded file name should end with {expected_extension} but got {local_file[-len(expected_extension):]}"

    # 3. Check that it's not empty
    assert os.path.getsize(local_file) > 0, "File should not be empty after download."


def test_save_base64_to_temp_file():
    # Example base64 data URI
    data_uri = (
        "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA"
        "AAAFCAYAAACNbyblAAAAHElEQVQI12P4"
        "//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="
    )

    # Call the function
    temp_file_path = uri_to_local_file_path(data_uri)

    # Check that the temp file was created
    assert os.path.exists(temp_file_path)

    # Check that the temp file contains some content
    with open(temp_file_path, "rb") as f:
        content = f.read()
        assert len(content) > 0

    # Clean up the temp file
    os.remove(temp_file_path)
