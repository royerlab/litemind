import json

from litemind.agent.messages.actions.action_base import ActionBase
from litemind.agent.messages.actions.tool_call import ToolCall
from litemind.agent.messages.actions.tool_use import ToolUse
from litemind.media.types.media_action import Action
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text


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
            if not block.has_type(Action) or (
                block.has_type(Action) and isinstance(block.media.action, ToolUse)
            ):
                # Reset the current_tool_call_formatted_message
                current_tool_call_formatted_message = None

            if block.has_type(Text):

                # Get strings:
                text: str = block.get_content()

                # Adding text block:
                formatted_message["content"].append({"type": "text", "text": text})

            elif block.has_type(Image):

                # Cast to Image:
                image: Image = block.media

                # non-local image URI:
                image_uri = image.to_remote_or_data_uri()

                # Add Image block:
                formatted_message["content"].append(
                    {"type": "image_url", "image_url": {"url": image_uri}}
                )

            elif block.has_type(Audio):

                # Cast to Image:
                audio: Audio = block.media

                # Non-local image URI:
                audio_uri = audio.to_remote_or_data_uri()

                # Media type:
                media_type = audio.get_media_type()

                # Audio block:
                formatted_message["content"].append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_uri,
                            "format": media_type.split("/")[1],
                        },
                    }
                )

            elif block.has_type(Action):

                tool_action: ActionBase = block.get_content()

                if isinstance(tool_action, ToolCall):

                    # Extract the tool use from the block
                    tool_call: ToolCall = tool_action

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

                elif isinstance(tool_action, ToolUse):

                    # Extract the tool use from the block
                    tool_use: ToolUse = tool_action

                    # Format the tool message for OpenAI:s
                    formatted_message["role"] = "tool"
                    formatted_message["tool_call_id"] = tool_use.id
                    formatted_message["content"] = tool_use.result

                else:
                    raise ValueError(
                        f"Unsupported action type: {type(tool_action).__name__}"
                    )

            else:
                raise ValueError(f"Unsupported block type: {block.get_type()}")

        # Append the formatted message to the list if it has been filled:
        if current_tool_call_formatted_message is None:
            openai_formatted_messages.append(formatted_message)

    return openai_formatted_messages
