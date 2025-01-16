from typing import List

from litemind.agent.message import Message
from litemind.apis.utils.dowload_image_to_tempfile import \
    download_image_to_temp_file
from litemind.apis.utils.get_media_type_from_uri import get_media_type_from_uri
from litemind.apis.utils.read_file_and_convert_to_base64 import \
    read_file_and_convert_to_base64
from litemind.apis.utils.transform_video_uris_to_images_and_audio import \
    transform_video_uris_to_images_and_video


def _convert_messages_for_anthropic(messages: List[Message]) -> List[
    'MessageParam']:
    """
    Convert litemind Messages into Anthropic's MessageParam format:
        [
          {"role": "user", "content": [{"type": "text", "text": "Hello!"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}]},
          {"role": "assistant", "content": [{"type": "text", "text": "Hello there!"}]},
          ...
        ]

    """

    # Initialize the list of messages:
    from anthropic.types import MessageParam
    anthropic_messages: List[MessageParam] = []

    # Iterate over each message:
    for message in messages:

        # Convert video URIs to images and audio because Ollama does not natively support videos:
        message = transform_video_uris_to_images_and_video(message)

        content = []

        # Append the text content:
        if message.text:
            content.append({"type": "text", "text": message.text})

        # Append the image content:
        for image_uri in message.image_uris:

            # Anthropic API requires the image to be in base64 format:

            if image_uri.startswith("data:image/"):
                # If it's a data URI, put the base64 data in base64_data:
                base64_data = image_uri.split(",")[-1]

            elif image_uri.startswith("http://") or image_uri.startswith(
                    "https://"):
                # If it's a remote URL, download the image to a temp file:
                local_path = download_image_to_temp_file(image_uri)

                # Convert the image to base64:
                base64_data = read_file_and_convert_to_base64(local_path)

            elif image_uri.startswith("file://"):
                # If it's a local file path, read the file and convert to base64:
                local_path = image_uri.replace("file://", "")
                base64_data = read_file_and_convert_to_base64(local_path)

            else:
                # raise exception:
                raise ValueError(
                    f"Invalid image URI: '{image_uri}' (must start with 'data:image/', 'http://', 'https://', or 'file://')")

            # Determine media type:
            media_type = get_media_type_from_uri(image_uri)

            # Append the image content to the message:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64_data,
                },
            })

        # Append the message to the list of messages:
        anthropic_messages.append({
            "role": message.role,
            "content": content,
        })
    return anthropic_messages
