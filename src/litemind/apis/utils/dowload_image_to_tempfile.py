import tempfile

import requests


def download_image_to_temp_file(url: str) -> str:
    """
    Downloads the image from an HTTP(S) URL and saves it to a temp file.
    Tries to infer extension from Content-Type or from the URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    }

    resp = requests.get(url, headers=headers)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "")
    if "image/png" in content_type:
        extension = ".png"
    elif "image/jpeg" in content_type:
        extension = ".jpg"
    elif "image/gif" in content_type:
        extension = ".gif"
    elif "image/webp" in content_type:
        extension = ".webp"
    else:
        urll = url.lower()
        if urll.endswith(".jpg") or urll.endswith(".jpeg"):
            extension = ".jpg"
        elif urll.endswith(".png"):
            extension = ".png"
        elif urll.endswith(".gif"):
            extension = ".gif"
        elif urll.endswith(".webp"):
            extension = ".webp"
        else:
            extension = ".png"

    with tempfile.NamedTemporaryFile(suffix=extension,
                                     delete=False) as tmp_file:
        tmp_file.write(resp.content)
        tmp_path = tmp_file.name

    return tmp_path
