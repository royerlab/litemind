from typing import Any, Optional, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.apis.utils.json_to_object import json_to_object


def process_response_from_gemini(
    gemini_response: Any,
    model_name: str,
    max_num_output_tokens: int,
    response_format: Optional[Union[BaseModel, str]] = None,
) -> Message:

    # Initialize the processed response:
    processed_reponse = Message(role="assistant")

    # Text content and tool calls:
    text_content, tool_cals = gemini_response["text"], gemini_response["tool_calls"]

    # If there is text content, add it to the processed response:
    if text_content:
        processed_reponse.append_text(text_content)

    # Variable to hold whether the response is a tool use:
    is_tool_use = False

    for tool_call in tool_cals:
        tool_name = tool_call["name"]
        arguments = tool_call.get("args", {})
        tool_call_id = tool_call.get("id", "")

        processed_reponse.append_tool_call(
            tool_name=tool_name, arguments=arguments, id=tool_call_id
        )
        is_tool_use = True

    if response_format and not is_tool_use:

        processed_reponse = json_to_object(
            processed_reponse, response_format, text_content
        )

    return processed_reponse
