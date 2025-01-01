import base64
import re
import tempfile
from typing import List, Dict, Any

import \
    requests  # Make sure 'requests' is in your environment, or handle differently

from litemind.agent.message import Message


@staticmethod
def _convert_messages_for_ollama(messages: List[Message]) -> List[
    Dict[str, Any]]:
    """
    Convert litemind 'Message' objects into Ollama's expected message format.

    If a message has image URLs:
      - If 'data:image/...' => decode Base64 => save to temp file
      - If 'http(s)://...'  => download => save to temp file
      - Else assume it is already a local file path.

    Return a list of dicts like:
      [
        {
          "role": "user",
          "content": "...",
          "images": ["/tmp/tmp1234.png", ...]
        },
        ...
      ]
    """

    def _save_base64_to_temp_file(data_uri: str) -> str:
        """
        Saves a data URI (data:image/...;base64,...) to a temporary file.
        Returns the local file path.
        """
        # Example: data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA...
        # Extract the MIME type
        match = re.match(r"data:image/(png|jpeg|jpg);base64,(.*)", data_uri,
                         re.IGNORECASE)
        if not match:
            # fallback: strip "data:image/..." or just call it ".png"
            extension = ".png"
            base64_part = data_uri.split(",", 1)[-1]
        else:
            image_type = match.group(1).lower()
            extension = ".png" if image_type == "png" else ".jpg"
            base64_part = match.group(2)

        # Decode from base64
        image_data = base64.b64decode(base64_part)

        # Save to a temp file
        with tempfile.NamedTemporaryFile(suffix=extension,
                                         delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name

        return tmp_path

    def _download_url_to_temp_file(url: str) -> str:
        """
        Downloads the image from an HTTP(S) URL and saves it to a temp file.
        Tries to infer extension from Content-Type or from the URL.
        """

        # Add a descriptive User-Agent to the headers:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()  # Raise an error if the download failed

        # Get extension from headers or fallback from URL or default to .png
        content_type = resp.headers.get("Content-Type", "")
        if "image/png" in content_type:
            extension = ".png"
        elif "image/jpeg" in content_type:
            extension = ".jpg"
        else:
            # Try to parse from the URL
            if url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
                extension = ".jpg"
            elif url.lower().endswith(".png"):
                extension = ".png"
            else:
                extension = ".png"

        # Save to a temp file
        with tempfile.NamedTemporaryFile(suffix=extension,
                                         delete=False) as tmp_file:
            tmp_file.write(resp.content)
            tmp_path = tmp_file.name

        return tmp_path

    ollama_messages = []

    for msg in messages:
        message_dict = {
            "role": msg.role,
            "content": msg.text,
        }

        local_image_paths = []
        for image_url in msg.image_uris:
            if image_url.startswith("data:image/"):
                # It's a data URI -> decode Base64 -> local file
                local_path = _save_base64_to_temp_file(image_url)
                local_image_paths.append(local_path)
            elif image_url.lower().startswith(
                    "http://") or image_url.lower().startswith("https://"):
                # It's a remote URL -> download -> local file
                local_path = _download_url_to_temp_file(image_url)
                local_image_paths.append(local_path)
            else:
                # Assume it's already a local file path
                local_image_paths.append(image_url)

        if local_image_paths:
            message_dict["images"] = local_image_paths

        ollama_messages.append(message_dict)

    return ollama_messages
