from typing import Optional


def get_media_type_from_uri(uri: str) -> Optional[str]:
    """
    Extract the media type from the given URI.

    Parameters
    ----------
    uri: str
        The URI to extract the media type from.

    Returns
    -------
    str
        The media type of the

    """

    if uri.startswith("data:"):
        return uri.split(":")[1].split(";")[0]
    elif (
        uri.startswith("http://")
        or uri.startswith("https://")
        or uri.startswith("file://")
    ):
        # Look at file extension:
        if uri.endswith(".png"):
            return "image/png"
        elif uri.endswith(".jpg") or uri.endswith(".jpeg"):
            return "image/jpeg"
        elif uri.endswith(".gif"):
            return "image/gif"
        elif uri.endswith(".webp"):
            return "image/webp"
        elif uri.endswith(".mp4"):
            return "video/mp4"
        elif uri.endswith(".webm"):
            return "video/webm"
        elif uri.endswith(".mov"):
            return "video/quicktime"
        elif uri.endswith(".avi"):
            return "video/x-msvideo"
        elif uri.endswith(".mkv"):
            return "video/x-matroska"
        elif uri.endswith(".mp3"):
            return "audio/mpeg"
        elif uri.endswith(".wav"):
            return "audio/wav"
        elif uri.endswith(".ogg"):
            return "audio/ogg"
        elif uri.endswith(".flac"):
            return "audio/flac"
        elif uri.endswith(".aac"):
            return "audio/aac"
        elif uri.endswith(".wma"):
            return "audio/x-ms-wma"
        elif uri.endswith(".m4a"):
            return "audio/mp4"
        elif uri.endswith(".opus"):
            return "audio/opus"
        elif uri.endswith(".pdf"):
            return "application/pdf"
        elif uri.endswith(".doc") or uri.endswith(".docx"):
            return "application/msword"
        elif uri.endswith(".xls") or uri.endswith(".xlsx"):
            return "application/vnd.ms-excel"
        elif uri.endswith(".ppt") or uri.endswith(".pptx"):
            return "application/vnd.ms-powerpoint"
        elif uri.endswith(".txt"):
            return "text/plain"
        elif uri.endswith(".html") or uri.endswith(".htm"):
            return "text/html"
        elif uri.endswith(".json"):
            return "application/json"
        elif uri.endswith(".xml"):
            return "application/xml"
        elif uri.endswith(".epub"):
            return "application/epub+zip"
        elif uri.endswith(".csv"):
            return "text/csv"
        elif uri.endswith(".tab"):
            return "text/tab-separated-values"
        else:
            return None
