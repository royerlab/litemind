from typing import List, Optional

from pydantic import BaseModel

from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType
from litemind.apis.utils.get_media_type_from_uri import get_media_type_from_uri
from litemind.apis.utils.read_file_and_convert_to_base64 import read_file_and_convert_to_base64, base64_to_data_uri
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


def convert_messages_for_anthropic(messages: List[Message],
                                   response_format: Optional[BaseModel] = None,
                                   cache_support: bool = True) -> List['MessageParam']:
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
                text: str = block.content

                if message.role == 'assistant':
                    # remove trailing whitespace as it is not allowed!
                    text = text.rstrip()

                content.append({"type": "text", "text": text})
            elif block.block_type == BlockType.Image:
                image_uri = block.content
                media_type = get_media_type_from_uri(image_uri)
                if image_uri.startswith("file://"):
                    local_path = image_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    image_uri = base64_to_data_uri(base64_data, media_type)
                elif image_uri.startswith("http://") or image_uri.startswith("https://"):
                    local_path = uri_to_local_file_path(image_uri)
                    base64_data = read_file_and_convert_to_base64(local_path)
                elif image_uri.startswith("data:image/"):
                    base64_data = image_uri.split(",")[-1]
                else:
                    raise ValueError(f"Unsupported image URI: {image_uri}")

                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": image_uri.split(",")[-1] if image_uri.startswith("data:image/") else base64_data,
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
                        "data": audio_uri.split(",")[-1] if audio_uri.startswith("data:audio/") else base64_data,
                    },
                })
            # Add more block types as needed

        anthropic_messages.append({
            "role": message.role,
            "content": content,
        })

    # Add cache_control to the last message if requested:
    if cache_support:
        if anthropic_messages and anthropic_messages[-1]['content']:
            anthropic_messages[-1]['content'][-1]['cache_control'] = {"type": "ephemeral"}

    # Add prompt that specifies that response should be in JSOn following given JSON schema:
    if response_format:
        anthropic_messages[-1]['content'].append({
            "type": "text",
            "text": f"\nPlease provide answer as a JSON string that adheres to the following schema:\n{response_format.model_json_schema()}\n\n Without text before or after.",
        })

    return anthropic_messages
