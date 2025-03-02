from typing import List, Optional

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.messages.tool_call import ToolCall
from litemind.agent.messages.tool_use import ToolUse
from litemind.apis.utils.get_media_type_from_uri import get_media_type_from_uri
from litemind.apis.utils.read_file_and_convert_to_base64 import (
    base64_to_data_uri,
    read_file_and_convert_to_base64,
)
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


def convert_messages_for_anthropic(
    messages: List[Message],
    response_format: Optional[BaseModel] = None,
    cache_support: bool = True,
) -> List["MessageParam"]:
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

                if message.role == "assistant":
                    # remove trailing whitespace as it is not allowed by Anthropic!
                    text = text.rstrip()

                content.append({"type": "text", "text": text})
            elif block.block_type == BlockType.Image:

                # Get the image URI and convert file to base64:
                image_uri = block.content
                media_type = get_media_type_from_uri(image_uri)
                if image_uri.startswith("file://"):
                    local_path = image_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    image_uri = base64_to_data_uri(base64_data, media_type)
                elif image_uri.startswith("http://") or image_uri.startswith(
                    "https://"
                ):
                    local_path = uri_to_local_file_path(image_uri)
                    base64_data = read_file_and_convert_to_base64(local_path)
                elif image_uri.startswith("data:image/"):
                    base64_data = image_uri.split(",")[-1]
                else:
                    raise ValueError(f"Unsupported image URI: {image_uri}")

                # Add the image to the content:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": (
                                image_uri.split(",")[-1]
                                if image_uri.startswith("data:image/")
                                else base64_data
                            ),
                        },
                    }
                )
            elif block.block_type == BlockType.Audio:

                # Get the audio URI and convert file to base64:
                audio_uri = block.content
                media_type = get_media_type_from_uri(audio_uri)
                if audio_uri.startswith("file://"):
                    local_path = audio_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    audio_uri = base64_to_data_uri(base64_data, media_type)

                # Add the audio to the content:
                content.append(
                    {
                        "type": "audio",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": (
                                audio_uri.split(",")[-1]
                                if audio_uri.startswith("data:audio/")
                                else base64_data
                            ),
                        },
                    }
                )
            elif block.block_type == BlockType.Document and block.content.endswith(
                ".pdf"
            ):
                # Convert PDF to base64:
                pdf_uri = block.content
                media_type = "application/pdf"
                if pdf_uri.startswith("file://"):
                    local_path = pdf_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    pdf_uri = base64_to_data_uri(base64_data, media_type)
                elif pdf_uri.startswith("http://") or pdf_uri.startswith("https://"):
                    local_path = uri_to_local_file_path(pdf_uri)
                    base64_data = read_file_and_convert_to_base64(local_path)
                elif pdf_uri.startswith("data:application/pdf;"):
                    base64_data = pdf_uri.split(",")[-1]
                else:
                    raise ValueError(f"Unsupported PDF URI: {pdf_uri}")

                # Add the document to the content:
                content.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": (
                                pdf_uri.split(",")[-1]
                                if pdf_uri.startswith("data:application/pdf;")
                                else base64_data
                            ),
                        },
                    }
                )
            elif block.block_type == BlockType.Tool:

                # if contents is a ToolUse object do the following:
                if isinstance(block.content, ToolUse):

                    # Get the tool use object:
                    tool_use: ToolUse = block.content

                    # Add tool use to the content:
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": [
                                {
                                    "type": "text",
                                    "text": tool_use.result,
                                }
                            ],
                        }
                    )

                elif isinstance(block.content, ToolCall):

                    # Get the tool use object:
                    tool_call: ToolUse = block.content

                    # Add the tool to the content:
                    content.append(
                        {
                            "id": tool_call.id,
                            "type": "tool_use",
                            "name": tool_call.tool_name,
                            "input": tool_call.arguments,
                        }
                    )

            else:
                raise ValueError(f"Unsupported block type: {block.block_type}")

        anthropic_messages.append(
            {
                "role": message.role,
                "content": content,
            }
        )

    # Add cache_control to the last message if requested:
    if cache_support:
        if anthropic_messages and anthropic_messages[-1]["content"]:
            anthropic_messages[-1]["content"][-1]["cache_control"] = {
                "type": "ephemeral"
            }

    # Add prompt that specifies that response should be in JSON following given JSON schema:
    if response_format:
        anthropic_messages[-1]["content"].append(
            {
                "type": "text",
                "text": f"\nPlease provide answer as a JSON string that adheres to the following schema:\n{response_format.model_json_schema()}\n\n Without text before or after.",
            }
        )

    return anthropic_messages
