import atexit
import base64
import binascii
import mimetypes
import os
import random
import shutil
import tempfile
import urllib.parse
from functools import lru_cache

import requests

from litemind.utils.uri_utils import is_uri, is_valid_path

# Keep track of temp directories for cleanup
_temp_directories = set()


def _register_cleanup():
    """Register cleanup of temporary directories at process exit."""

    def cleanup_temp_dirs():
        for dir_path in _temp_directories:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path, ignore_errors=True)

    atexit.register(cleanup_temp_dirs)


# Register cleanup on module import
_register_cleanup()


@lru_cache
def uri_to_local_file_path(file_uri: str) -> str:
    """
    Given a file URI (which can be an existing local file path, a remote URL, or
    Base64-encoded data), returns the absolute path to a local file.

    1. If file_uri is already a local file path, returns its absolute path.
    2. If file_uri is a 'file://' URI, converts it to a local path.
    3. If file_uri is a remote URL (http/https/ftp), downloads it to a temp file.
    4. If file_uri is a data URI or raw Base64, decodes it to a temp file.

    Parameters
    ----------
    file_uri : str
        The file URI to be converted, or a well formed local file path.

    Returns
    -------
    str
        The absolute path to the local file.
    """
    if not file_uri:
        raise ValueError("Empty or null file URI provided")

    # SCENARIO 0: Check if it's already a valid local file path
    if is_valid_path(file_uri) and os.path.exists(file_uri):
        # If it's a valid local path, return its absolute path
        return os.path.abspath(file_uri)

    # Parse the URI to determine its type
    parsed = urllib.parse.urlparse(file_uri)

    # SCENARIO 1: file:// URI
    if parsed.scheme == "file":
        local_path = urllib.parse.unquote(parsed.path)
        # On Windows, remove leading slash if present
        if os.name == "nt" and local_path.startswith("/"):
            local_path = local_path[1:]
        local_path = os.path.abspath(local_path)
        if not os.path.exists(local_path):
            raise ValueError(f"Local file not found: '{local_path}'")
        return local_path

    # List of possible user agents
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:102.0) Gecko/20100101 Firefox/102.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.3 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Mobile Safari/537.36",
        "Mozilla/5.0 (iPhone; CPU iPhone OS 15_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.4 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:98.0) Gecko/20100101 Firefox/98.0",
        "Mozilla/5.0 (iPad; CPU OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1",
        "Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_4_0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
    ]

    # SCENARIO 2: Remote URL (http/https/ftp)
    if parsed.scheme in ["http", "https", "ftp"]:
        selected_user_agent = random.choice(user_agents)
        headers = {"User-Agent": selected_user_agent}

        try:
            with requests.get(file_uri, headers=headers, stream=True) as r:
                r.raise_for_status()

                # Derive filename from the URL path or Content-Type
                filename = os.path.basename(parsed.path)
                if not filename or "." not in filename:
                    content_type = r.headers.get("Content-Type", "")
                    guessed_ext = mimetypes.guess_extension(content_type.split(";")[0])
                    if guessed_ext is None:
                        guessed_ext = ".bin"
                    if not filename:
                        filename = f"download{guessed_ext}"
                    else:
                        filename += guessed_ext

                # Create a temp directory, store the file there
                temp_dir = tempfile.mkdtemp(prefix="download_")
                _temp_directories.add(temp_dir)  # For cleanup later
                local_path = os.path.join(temp_dir, filename)

                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)

            return os.path.abspath(local_path)
        except requests.exceptions.RequestException as e:
            raise ValueError(
                f"Failed to download from URL: {file_uri}. Error: {str(e)}"
            )

    # SCENARIO 3: data: URI
    mime_type = None
    b64_data = file_uri

    if parsed.scheme == "data":
        if "," not in file_uri:
            raise ValueError(f"Invalid data URI format: {file_uri}")

        header, b64_data = file_uri.split(",", 1)
        # Extract MIME type from header
        if ";" in header:
            mime_type = header.split(":", 1)[1].split(";", 1)[0]

    # If it's not a recognized URI scheme, try to interpret as Base64
    elif not is_uri(file_uri):
        # Try to validate if it looks like base64
        try:
            # Check if padding is correct
            padding_needed = len(b64_data) % 4
            if padding_needed:
                b64_data += "=" * (4 - padding_needed)

            # Verify it can be decoded
            base64.b64decode(b64_data, validate=True)
        except (binascii.Error, ValueError) as e:
            raise ValueError(
                f"Invalid URI format and not valid Base64 data: '{file_uri}'. Error: {str(e)}"
            )
    else:
        raise ValueError(
            f"Unsupported URI scheme: '{parsed.scheme}'. Supported schemes: file, http, https, ftp, data"
        )

    # Decode base64 data
    try:
        file_data = base64.b64decode(b64_data)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Failed to decode Base64 data: {str(e)}")

    # Decide on a filename with appropriate extension
    ext = None
    if mime_type:
        ext = mimetypes.guess_extension(mime_type)
    if not ext:
        ext = ".bin"

    temp_dir = tempfile.mkdtemp(prefix="b64_")
    _temp_directories.add(temp_dir)  # For cleanup later
    filename = f"decoded{ext}"
    local_path = os.path.join(temp_dir, filename)

    with open(local_path, "wb") as f:
        f.write(file_data)

    return os.path.abspath(local_path)
