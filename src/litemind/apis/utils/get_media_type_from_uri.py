def get_media_type_from_uri(uri: str) -> str:
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

    if uri.startswith("data:image/"):
        return uri.split(":")[1].split(";")[0]
    elif uri.startswith("http://") or uri.startswith(
            "https://") or uri.startswith("file://"):
        # Look at file extension:
        if uri.endswith(".png"):
            return "image/png"
        elif uri.endswith(".jpg") or uri.endswith(".jpeg"):
            return "image/jpeg"
        elif uri.endswith(".gif"):
            return "image/gif"
        elif uri.endswith(".webp"):
            return "image/webp"
        else:
            return None
