from typing import Any, Optional, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.utils.json_to_object import json_to_object


def process_response_from_openai(
    openai_response: Any, response_format: Optional[Union[BaseModel, str]]
) -> Message:
    """
    Process OpenAI response, checking for function calls and executing them if needed.

    Parameters
    ----------
    openai_response : Any
        The response from OpenAI API.
    response_format : Optional[BaseModel | str]
        The format of the response.
    thinking_model : bool
        Whether the model is a thinking model or not.

    Returns
    -------
    Message
        The response message to return.

    """

    processed_response = Message(role="assistant")

    openai_message = openai_response.choices[0].message

    if openai_message.content is not None:
        processed_response.append_text(openai_message.content)
        text_content = openai_message.content
    else:
        text_content = None

    if openai_message.tool_calls:
        for tool_call in openai_message.tool_calls:
            processed_response.append_tool_call(
                tool_name=tool_call.function.name,
                arguments=tool_call.function.parsed_arguments,
                id=tool_call.id,
            )

    if response_format and text_content is not None:
        processed_response = json_to_object(
            processed_response, response_format, text_content
        )

    return processed_response
