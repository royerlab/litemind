import tempfile

import requests


def download_document_to_temp_file(url: str) -> str:
    """
    Downloads the document from an HTTP(S) URL and saves it to a temp file.
    Tries to infer extension from Content-Type or from the URL.
    """

    # If file is already local, just return the path:
    if url.startswith("file://"):
        return url.replace("file://", "")

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "application/pdf" in content_type:
        extension = ".pdf"
    elif "application/epub+zip" in content_type:
        extension = ".epub"
    elif "text/html" in content_type:
        extension = ".html"
    elif "text/txt" in content_type:
        extension = ".txt"
    else:
        urll = url.lower()
        if urll.endswith(".pdf"):
            extension = ".pdf"
        elif urll.endswith(".epub"):
            extension = ".epub"
        elif urll.endswith(".html"):
            extension = ".html"
        elif urll.endswith(".txt"):
            extension = ".txt"
        else:
            raise ValueError("Unknown document type")

    with tempfile.NamedTemporaryFile(suffix=extension,
                                     delete=False) as tmp_file:
        tmp_file.write(resp.content)
        tmp_path = tmp_file.name

    return tmp_path
