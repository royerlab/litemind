from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.messages.tool_call import ToolCall
from litemind.agent.messages.tool_use import ToolUse
from litemind.apis.utils.convert_image_to_png import convert_image_to_png
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


def convert_messages_for_ollama(
    messages: List[Message], response_format: Optional[BaseModel] = None
) -> List[Dict[str, Any]]:
    """
    Convert litemind 'Message' objects into Ollama's expected message format.

    """

    # Initialize the list of messages:
    ollama_messages = []

    # Iterate over each message:
    for message in messages:

        # Iterate over each block in the message
        for block in message.blocks:

            # Initialize the message dictionary:
            message_dict = {"role": message.role, "content": ""}

            if block.block_type == BlockType.Text:
                if block.content.strip():
                    # Append the text content to the message dictionary:
                    message_dict["content"] += block.content + "\n"

                    # Append the message to the list of messages:
                    ollama_messages.append(message_dict)

            elif block.block_type == BlockType.Image:
                # Get the image URI:
                image_uri = block.content

                # Convert the image URI to a local file path:
                local_image_path = uri_to_local_file_path(image_uri)

                if local_image_path.endswith(".webp"):
                    local_image_path = convert_image_to_png(local_image_path)

                # Append the local image paths to the message dictionary:
                if local_image_path:
                    message_dict["images"] = [local_image_path]

                # Append the message to the list of messages:
                ollama_messages.append(message_dict)

            elif block.block_type == BlockType.Tool:

                # if contents is a ToolUse object do the following:
                if isinstance(block.content, ToolUse):

                    # Get the tool use object:
                    tool_use: ToolUse = block.content

                    # Set the role to 'tool':
                    message_dict["role"] = "tool"

                    # Append the tool use results to the message dictionary:
                    message_dict["content"] = tool_use.pretty_string()

                elif isinstance(block.content, ToolCall):

                    # Get the tool use object:
                    tool_call: ToolUse = block.content

                    # Initialize the tool calls list:
                    tool_calls = []

                    # Append the tool call to the tool calls list:
                    tool_calls.append(
                        {
                            "function": {
                                "name": tool_call.tool_name,
                                "arguments": tool_call.arguments,
                            },
                            "type": "function_call",
                        }
                    )

                    # Set the tool calls in the message dictionary:
                    message_dict["tool_calls"] = tool_calls

                # Append the message to the list of messages:
                ollama_messages.append(message_dict)

            else:
                raise ValueError(f"Block type {block.block_type} not supported")

    if response_format:

        # Create a message dictionary for the response format:
        message_dict = {
            "role": "user",
            "content": f"Provide the answer in JSON adhering to the following schema:\n{response_format.model_json_schema()}\n",
        }

        # Appends the message to the list of messages:
        ollama_messages.append(message_dict)

    return ollama_messages
