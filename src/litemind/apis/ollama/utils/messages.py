from typing import List, Dict, Any

from litemind.agent.message import Message
from litemind.apis.utils.convert_image_to_png import convert_image_to_png
from litemind.apis.utils.dowload_image_to_tempfile import \
    download_image_to_temp_file
from litemind.apis.utils.write_base64_to_temp_file import \
    write_base64_to_temp_file


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

    # Initialize the list of messages:
    ollama_messages = []

    # Iterate over each message:
    for msg in messages:

        # Initialize the message dictionary:
        message_dict = {
            "role": msg.role,
            "content": msg.text,
        }

        # Initialize the list of local image paths:
        local_image_paths = []

        # Append the image content:
        for image_url in msg.image_uris:

            if image_url.startswith("data:image/"):
                # It's a data URI -> decode Base64 -> local file
                local_path = write_base64_to_temp_file(image_url)
                local_image_paths.append(local_path)

            elif image_url.lower().startswith(
                    "http://") or image_url.lower().startswith("https://"):
                # It's a remote URL -> download -> local file
                local_path = download_image_to_temp_file(image_url)
                local_image_paths.append(local_path)

            elif image_url.startswith("file://"):
                # If it is a local file path
                # First remove the 'file://' prefix:
                image_url = image_url.replace("file://", "")

                # Then append the image URI to the list of local image paths:
                local_image_paths.append(image_url)

            else:
                # Raise an error if the image URI is invalid:
                raise ValueError(f"Invalid image URI: '{image_url}'")


        # If any image is in webp format, which is not supported by Ollama, convert it to PNG:
        for i, image_path in enumerate(local_image_paths):
            if image_path.endswith(".webp"):
                local_image_paths[i] = convert_image_to_png(image_path)

        # Append the local image paths to the message dictionary:
        if local_image_paths:
            message_dict["images"] = local_image_paths

        # Append the message to the list of messages:
        ollama_messages.append(message_dict)

    return ollama_messages
