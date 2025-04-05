from typing import Any, Optional, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.utils.json_to_object import json_to_object


def process_response_from_anthropic(
    anthropic_response: Any,
    response_format: Optional[Union[BaseModel, str]] = None,
) -> Message:
    """
    Process Anthropic response, checking for function calls and executing them if needed.

    Parameters
    ----------
    anthropic_response : Any
        The response from Anthropic API.
    response_format : Optional[BaseModel | str]
        The format of the response.

    Returns
    -------
    Message
        The response message to return.

    """

    # Initialize the processed response:
    processed_reponse = Message(role="assistant")

    # Text content:
    text_content = ""

    # Variable to hold whether the response is a tool use:
    is_tool_use = False

    # Create a litemind response message from Anthropic's response that includes tool calls:
    for block in anthropic_response.content:
        if block.type == "text":
            processed_reponse.append_text(block.text)
            text_content += block.text
        elif block.type == "tool_use":
            processed_reponse.append_tool_call(
                tool_name=block.name, arguments=block.input, id=block.id
            )
            is_tool_use = True
        elif block.type == "thinking":
            processed_reponse.append_thinking(block.thinking, signature=block.signature)
        elif block.type == "redacted_thinking":
            processed_reponse.append_thinking(block.data, redacted=True)
        else:
            raise ValueError(f"Unexpected block type: {block.type}")

    if response_format and not is_tool_use:
        processed_reponse = json_to_object(
            processed_reponse, response_format, text_content
        )

    return processed_reponse
