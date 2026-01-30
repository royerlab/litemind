from typing import Any, Optional, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.utils.extract_thinking import extract_thinking_content
from litemind.utils.json_to_object import json_to_object


def process_response_from_gemini(
    gemini_response: Any,
    response_format: Optional[Union[BaseModel, str]] = None,
) -> Message:
    """
    Process a Gemini response dictionary into a Message object.

    Parameters
    ----------
    gemini_response : Any
        Response dictionary from aggregate_chat_response with keys:
        - "text": The text content
        - "thinking": Native thinking content (if any)
        - "tool_calls": List of function calls
    response_format : Optional[Union[BaseModel, str]]
        Optional response format for structured output

    Returns
    -------
    Message
        Processed response as a Message object
    """
    # Initialize the processed response:
    processed_response = Message(role="assistant")

    # Text content and tool calls:
    text_content = gemini_response.get("text", "")
    tool_calls = gemini_response.get("tool_calls", [])

    # Check for native thinking content from the new SDK
    native_thinking = gemini_response.get("thinking")

    if native_thinking:
        # Use native thinking content from the SDK
        processed_response.append_thinking(native_thinking)
    else:
        # Fall back to extracting thinking from text (for backward compatibility)
        text_content, thinking_content = extract_thinking_content(text_content)

        # If there is thinking content, add it to the processed response:
        if thinking_content:
            processed_response.append_thinking(thinking_content)

    # If there is text content, add it to the processed response:
    if text_content:
        processed_response.append_text(text_content)

    # Variable to hold whether the response is a tool use:
    is_tool_use = False

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        arguments = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        processed_response.append_tool_call(
            tool_name=tool_name, arguments=arguments, id=tool_call_id
        )
        is_tool_use = True

    if response_format and not is_tool_use:
        processed_response = json_to_object(
            processed_response, response_format, text_content
        )

    return processed_response
