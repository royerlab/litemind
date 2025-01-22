import os
import base64
import random
import requests
import tempfile
import urllib.parse

def uri_to_local_file_path(file_uri: str) -> str:
    """
    Given a file URI (which can be an existing local file path, a remote URL, or
    a Base64-encoded file), returns the absolute path to a local file.

    If `file_uri` is already a local file, returns its absolute path.
    Otherwise, downloads (or decodes) the data into a temporary file
    and returns that temporary file's absolute path.

    Randomizes only the 'User-Agent' among 10 standard ones for remote downloads.
    """

    USER_AGENTS = [
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
    selected_user_agent = random.choice(USER_AGENTS)

    # We only set the 'User-Agent', everything else is minimal or default.
    headers = {
        "User-Agent": selected_user_agent
    }

    # Try interpreting as a file:// URI
    parsed = urllib.parse.urlparse(file_uri)

    # Case 1: If the scheme is 'file', parse out the local path
    if parsed.scheme == 'file':
        local_path = os.path.abspath(parsed.path)
        if os.path.exists(local_path):
            return local_path

    # Case 2: Check if file_uri is an existing local path
    if os.path.exists(file_uri):
        return os.path.abspath(file_uri)

    # Case 3: Check if it's a remote URL (http, https, ftp)
    if parsed.scheme in ['http', 'https', 'ftp']:
        with requests.get(file_uri, headers=headers, stream=True) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
            return os.path.abspath(tmp_file.name)

    # Case 4: Otherwise, assume Base64 data
    # Handle data URIs (e.g. 'data:application/pdf;base64,...')
    if file_uri.startswith('data:'):
        header, b64_data = file_uri.split(',', 1)
        file_data = base64.b64decode(b64_data)
    else:
        file_data = base64.b64decode(file_uri)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(file_data)
    return os.path.abspath(tmp_file.name)
