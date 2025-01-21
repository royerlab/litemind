from typing import List

from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType
from litemind.apis.utils.get_media_type_from_uri import get_media_type_from_uri
from litemind.apis.utils.read_file_and_convert_to_base64 import \
    read_file_and_convert_to_base64, base64_to_data_uri
from litemind.utils.dowload_image_to_tempfile import download_image_to_temp_file


def convert_messages_for_anthropic(messages: List[Message]) -> List[
    'MessageParam']:
    """
    Convert litemind Messages into Anthropic's MessageParam format:
        [
          {"role": "user", "content": [{"type": "text", "text": "Hello!"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}]},
          {"role": "assistant", "content": [{"type": "text", "text": "Hello there!"}]},
          ...
        ]
    """

    from anthropic.types import MessageParam
    anthropic_messages: List[MessageParam] = []

    for message in messages:
        content = []

        for block in message.blocks:
            if block.block_type == BlockType.Text:
                content.append({"type": "text", "text": block.content})
            elif block.block_type == BlockType.Image:
                image_uri = block.content
                media_type = get_media_type_from_uri(image_uri)
                if image_uri.startswith("file://"):
                    local_path = image_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    image_uri = base64_to_data_uri(base64_data, media_type)
                elif image_uri.startswith("http://") or image_uri.startswith(
                        "https://"):
                    local_path = download_image_to_temp_file(image_uri)
                    base64_data = read_file_and_convert_to_base64(local_path)
                elif image_uri.startswith("data:image/"):
                    base64_data = image_uri.split(",")[-1]

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_uri.split(",")[
                            -1] if image_uri.startswith(
                            "data:image/") else base64_data,
                    },
                })
            elif block.block_type == BlockType.Audio:
                audio_uri = block.content
                media_type = get_media_type_from_uri(audio_uri)
                if audio_uri.startswith("file://"):
                    local_path = audio_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    audio_uri = base64_to_data_uri(base64_data, media_type)
                content.append({
                    "type": "audio",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": audio_uri.split(",")[
                            -1] if audio_uri.startswith(
                            "data:audio/") else base64_data,
                    },
                })
            # Add more block types as needed

        anthropic_messages.append({
            "role": message.role,
            "content": content,
        })

    return anthropic_messages
