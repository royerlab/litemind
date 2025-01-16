import tempfile

import requests


def download_audio_to_temp_file(url: str) -> str:
    """
    Downloads the audio from an HTTP(S) URL and saves it to a temp file.
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
    if "audio/mpeg" in content_type:
        extension = ".mp3"
    elif "audio/wav" in content_type:
        extension = ".wav"
    elif "audio/aiff" in content_type:
        extension = ".aiff"
    elif "audio/aac" in content_type:
        extension = ".aac"
    elif "audio/ogg" in content_type:
        extension = ".ogg"
    elif "audio/flac" in content_type:
        extension = ".flac"
    else:
        urll = url.lower()
        if urll.endswith(".mp3"):
            extension = ".mp3"
        elif urll.endswith(".wav"):
            extension = ".wav"
        elif urll.endswith(".aiff"):
            extension = ".aiff"
        elif urll.endswith(".aac"):
            extension = ".aac"
        elif urll.endswith(".ogg"):
            extension = ".ogg"
        elif urll.endswith(".flac"):
            extension = ".flac"
        else:
            extension = ".mp3"

    with tempfile.NamedTemporaryFile(suffix=extension,
                                     delete=False) as tmp_file:
        tmp_file.write(resp.content)
        tmp_path = tmp_file.name

    return tmp_path
