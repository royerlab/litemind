import base64
import re
import tempfile
from typing import List

import requests
from anthropic.types import MessageParam

from litemind.agent.message import Message


def _convert_messages_for_anthropic(messages: List[Message]) -> List[
    MessageParam]:
    """
    Convert litemind Messages into Anthropic's MessageParam format:
        [
          {"role": "user", "content": [{"type": "text", "text": "Hello!"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}]},
          {"role": "assistant", "content": [{"type": "text", "text": "Hello there!"}]},
          ...
        ]

    If a Message contains images or structured content, you may adapt accordingly.
    """

    def _download_image_to_temp_file(url: str) -> str:
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
        else:
            if url.lower().endswith(".jpg") or url.lower().endswith(".jpeg"):
                extension = ".jpg"
            elif url.lower().endswith(".png"):
                extension = ".png"
            else:
                extension = ".png"

        with tempfile.NamedTemporaryFile(suffix=extension,
                                         delete=False) as tmp_file:
            tmp_file.write(resp.content)
            tmp_path = tmp_file.name

        return tmp_path

    def _save_base64_to_temp_file(data_uri: str) -> str:
        """
        Saves a data URI (data:image/...;base64,...) to a temporary file.
        Returns the local file path.
        """
        match = re.match(r"data:image/(png|jpeg|jpg);base64,(.*)", data_uri,
                         re.IGNORECASE)
        if not match:
            raise ValueError("Invalid data URI format")

        image_type = match.group(1).lower()
        extension = ".png" if image_type == "png" else ".jpg"
        base64_part = match.group(2)

        image_data = base64.b64decode(base64_part)

        with tempfile.NamedTemporaryFile(suffix=extension,
                                         delete=False) as tmp_file:
            tmp_file.write(image_data)
            tmp_path = tmp_file.name

        return tmp_path

    anthropic_messages: List[MessageParam] = []
    for msg in messages:
        content = []
        if msg.text:
            content.append({"type": "text", "text": msg.text})
        for image_url in msg.image_uris:
            if image_url.startswith("data:image/"):
                local_path = _save_base64_to_temp_file(image_url)
            elif image_url.startswith("http://") or image_url.startswith(
                    "https://"):
                local_path = _download_image_to_temp_file(image_url)
            else:
                local_path = image_url

            media_type = "image/png" if local_path.endswith(
                ".png") else "image/jpeg"
            with open(local_path, "rb") as image_file:
                base64_data = base64.b64encode(image_file.read()).decode(
                    "utf-8")

            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            })
        anthropic_messages.append({
            "role": msg.role,
            "content": content,
        })
    return anthropic_messages
