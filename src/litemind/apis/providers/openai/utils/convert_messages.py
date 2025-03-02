import json

from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.messages.tool_call import ToolCall
from litemind.agent.messages.tool_use import ToolUse
from litemind.apis.utils.get_media_type_from_uri import get_media_type_from_uri
from litemind.apis.utils.read_file_and_convert_to_base64 import (
    base64_to_data_uri,
    read_file_and_convert_to_base64,
)


def convert_messages_for_openai(messages):
    """
    Convert a list of Message objects into a format compatible with OpenAI's API.

    Parameters
    ----------
    messages : list of Message
        The list of Message objects to convert.

    Returns
    -------
    list of dict
        A list of dictionaries formatted for OpenAI's API.
    """

    # Initialize the list of formatted messages:
    openai_formatted_messages = []

    # Keep the current tool call formatted message:
    current_tool_call_formatted_message = None

    # Iterate over each message:
    for message in messages:

        # Start with the role of the message
        formatted_message = {"role": message.role, "content": []}

        # Iterate over each block in the message
        for block in message.blocks:

            # If the block is not a tool call block then set current_tool_call_formatted_message to None:
            if block.block_type != BlockType.Tool or (
                block.block_type == BlockType.Tool
                and isinstance(block.content, ToolUse)
            ):
                # Reset the current_tool_call_formatted_message
                current_tool_call_formatted_message = None

            if block.block_type == BlockType.Text:
                formatted_message["content"].append(
                    {"type": "text", "text": block.content}
                )
            elif block.block_type == BlockType.Image:
                image_uri = block.content
                media_type = get_media_type_from_uri(image_uri)
                if image_uri.startswith("file://"):
                    local_path = image_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    image_uri = base64_to_data_uri(base64_data, media_type)
                formatted_message["content"].append(
                    {"type": "image_url", "image_url": {"url": image_uri}}
                )
            elif block.block_type == BlockType.Audio:
                audio_uri = block.content
                media_type = get_media_type_from_uri(audio_uri)
                if audio_uri.startswith("file://"):
                    local_path = audio_uri.replace("file://", "")
                    base64_data = read_file_and_convert_to_base64(local_path)
                    audio_uri = base64_to_data_uri(base64_data, media_type)
                formatted_message["content"].append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_uri,
                            "format": media_type.split("/")[1],
                        },
                    }
                )
            elif block.block_type == BlockType.Tool:
                if isinstance(block.content, ToolCall):

                    # Extract the tool use from the block
                    tool_call: ToolCall = block.content

                    # If the current tool call is None then we are starting a new tool calls message:
                    if current_tool_call_formatted_message is None:
                        current_tool_call_formatted_message = formatted_message
                        del current_tool_call_formatted_message["content"]
                        current_tool_call_formatted_message["tool_calls"] = []
                        openai_formatted_messages.append(
                            current_tool_call_formatted_message
                        )

                    # Add the tool call to the tool_calls field of the previously formatted message:

                    # Convert tool_call.arguments dict to JSON:
                    oa_arguments = json.dumps(tool_call.arguments)

                    # Format the function  for OpenAI's API:
                    oa_function = {
                        "name": tool_call.tool_name,
                        "arguments": oa_arguments,
                    }

                    # Format the tool call for OpenAI's API:
                    oa_tool_call = {
                        "function": oa_function,
                        "type": "function",
                        "id": tool_call.id,
                    }

                    # Extract the tool call from the block
                    current_tool_call_formatted_message["tool_calls"].append(
                        oa_tool_call
                    )

                elif isinstance(block.content, ToolUse):

                    # Extract the tool use from the block
                    tool_use: ToolUse = block.content

                    # Format the tool message for OpenAI:s
                    formatted_message["role"] = "tool"
                    formatted_message["tool_call_id"] = tool_use.id
                    formatted_message["content"] = tool_use.result

            else:
                raise ValueError(f"Unsupported block type: {block.block_type}")

        # Append the formatted message to the list if it has been filled:
        if current_tool_call_formatted_message is None:
            openai_formatted_messages.append(formatted_message)

    return openai_formatted_messages
