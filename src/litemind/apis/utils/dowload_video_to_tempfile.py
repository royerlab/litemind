import tempfile

import requests


def download_video_to_temp_file(url: str) -> str:
    """
    Downloads the video from an HTTP(S) URL and saves it to a temp file.
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
    if "video/mp4" in content_type:
        extension = ".mp4"
    elif "video/webm" in content_type:
        extension = ".webm"
    elif "video/ogg" in content_type:
        extension = ".ogv"
    else:
        urll = url.lower()
        if urll.endswith(".mp4"):
            extension = ".mp4"
        elif urll.endswith(".webm"):
            extension = ".webm"
        elif urll.endswith(".ogv"):
            extension = ".ogv"
        else:
            extension = ".mp4"

    with tempfile.NamedTemporaryFile(suffix=extension,
                                     delete=False) as tmp_file:
        tmp_file.write(resp.content)
        tmp_path = tmp_file.name

    return tmp_path
