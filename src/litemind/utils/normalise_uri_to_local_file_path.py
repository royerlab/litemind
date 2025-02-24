import base64
import mimetypes
import os
import random
import tempfile
import urllib.parse

import requests


def uri_to_local_file_path(file_uri: str) -> str:
    """
    Given a file URI (which can be an existing local file path, a remote URL, or
    Base64-encoded data), returns the absolute path to a local file.

    1. If file_uri is already a local file (including 'file://' URI),
       returns its absolute path (scenario i).
    2. Otherwise (remote or base64), downloads/decodes into a temporary directory
       and returns the path to that file, using the best-guessed name/extension.
    """

    # List of possible user agents (as before).
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
    selected_user_agent = random.choice(user_agents)
    headers = {"User-Agent": selected_user_agent}

    parsed = urllib.parse.urlparse(file_uri)

    # ---------- SCENARIO 1: If it's a local file, return its absolute path ----------
    # 1(a). file:// URI
    if parsed.scheme == "file":
        local_path = os.path.abspath(parsed.path)
        if os.path.exists(local_path):
            return local_path

    # 1(b). Existing local file (absolute or relative path)
    if os.path.exists(file_uri):
        return os.path.abspath(file_uri)

    # ---------- SCENARIO 2: If it's remote, download to a temp directory ----------
    if parsed.scheme in ["http", "https", "ftp"]:
        with requests.get(file_uri, headers=headers, stream=True) as r:
            r.raise_for_status()

            # Attempt to derive filename from the URL path
            filename = os.path.basename(parsed.path)
            # If there's no filename or no extension in the URL, we might check Content-Type
            if not filename or "." not in filename:
                content_type = r.headers.get("Content-Type", "")
                # Attempt to guess an extension from the content-type
                guessed_ext = mimetypes.guess_extension(content_type.split(";")[0])
                if guessed_ext is None:
                    guessed_ext = ".bin"
                if not filename:
                    # Just pick something like "download.bin"
                    filename = f"download{guessed_ext}"
                else:
                    # e.g., we got 'somefile' from URL but no extension -> add the guessed_ext
                    filename += guessed_ext

            # Create a temp directory, store the file there
            temp_dir = tempfile.mkdtemp(prefix="download_")
            local_path = os.path.join(temp_dir, filename)

            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        return os.path.abspath(local_path)

    # ---------- SCENARIO 3: Otherwise, treat as Base64 data ----------
    # We try to guess a filename/extension, especially if it's data:... with a MIME type
    mime_type = None
    b64_data = file_uri
    if file_uri.startswith("data:"):
        # e.g., data:application/pdf;base64,JVBERi0xLjQK...
        header, b64_data = file_uri.split(",", 1)
        # header might be 'data:application/pdf;base64'
        # Extract 'application/pdf'
        if ";" in header:
            # e.g. 'data:application/pdf;base64'
            mime_type = header.split(":", 1)[1].split(";", 1)[0]  # 'application/pdf'

    # If it is none of these cases the raise exception:
    if not b64_data:
        raise ValueError(
            f"Invalid video URI: '{file_uri}' (must start with 'data:video/', 'http://', 'https://', 'ftp://', or 'file://')"
        )

    # Decode base64
    file_data = base64.b64decode(b64_data)

    # Decide on a filename for the base64 content:
    ext = None
    if mime_type:
        ext = mimetypes.guess_extension(mime_type)
    if not ext:
        ext = ".bin"

    # We'll just use something like 'decodedXXXX.bin'
    temp_dir = tempfile.mkdtemp(prefix="b64_")
    filename = f"decoded{ext}"
    local_path = os.path.join(temp_dir, filename)

    with open(local_path, "wb") as f:
        f.write(file_data)

    return os.path.abspath(local_path)
