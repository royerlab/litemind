from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from litemind.agent.messages.actions.action_base import ActionBase
from litemind.agent.messages.actions.tool_call import ToolCall
from litemind.agent.messages.actions.tool_use import ToolUse
from litemind.agent.messages.message import Message
from litemind.media.types.media_action import Action
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text


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

            if block.has_type(Text) and not block.is_thinking():

                # Get the Text media from the block:
                text_media: Text = block.media

                # Get Text's string:
                text: str = text_media.text

                if text.strip():
                    # Append the text content to the message dictionary:
                    message_dict["content"] += block.get_content() + "\n"

                    # Append the message to the list of messages:
                    ollama_messages.append(message_dict)

            elif block.has_type(Text) and block.is_thinking():
                # By default we don't send back the thinking content back to the LLM:
                pass

            elif block.has_type(Image):

                # Get the Image media from the block:
                image: Image = block.media

                # Convert the image URI to a local file path:
                local_image_path = image.to_local_file_path()

                if image.get_extension() == "webp":
                    # Convert the image to PNG format:
                    local_image_path = image.normalise_to_png()

                # Append the local image paths to the message dictionary:
                if local_image_path:
                    message_dict["images"] = [local_image_path]

                # Append the message to the list of messages:
                ollama_messages.append(message_dict)

            elif block.has_type(Action):

                tool_action: ActionBase = block.get_content()

                # if contents is a ToolUse object do the following:
                if isinstance(tool_action, ToolUse):

                    # Get the tool use object:
                    tool_use: ToolUse = tool_action

                    # Set the role to 'tool':
                    message_dict["role"] = "tool"

                    # Append the tool use results to the message dictionary:
                    message_dict["content"] = tool_use.pretty_string()

                elif isinstance(tool_action, ToolCall):

                    # Get the tool use object:
                    tool_call: ToolCall = tool_action

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

                else:
                    raise ValueError(
                        f"Unsupported action type: {type(tool_action).__name__}"
                    )

                # Append the message to the list of messages:
                ollama_messages.append(message_dict)

            else:
                raise ValueError(f"Block type {block.get_type_name()} not supported")

    if response_format:
        # Create a message dictionary for the response format:
        message_dict = {
            "role": "user",
            "content": f"Provide the answer in JSON adhering to the following schema:\n{response_format.model_json_schema()}\n",
        }

        # Appends the message to the list of messages:
        ollama_messages.append(message_dict)

    return ollama_messages
