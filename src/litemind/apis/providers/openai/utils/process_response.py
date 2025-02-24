from typing import Any, Optional

from pydantic import BaseModel

from litemind.agent.messages.message import Message


def process_response_from_openai(
    response: Any, response_format: Optional[BaseModel | str]
) -> Message:
    """
    Process OpenAI response, checking for function calls and executing them if needed.

    Parameters
    ----------
    response : Any
        The response from OpenAI API.
    response_format : Optional[BaseModel | str]
        The format of the response.

    Returns
    -------
    Message
        The response message to return.

    """

    # Extract the response content:
    response = response.choices[0].message.content

    # If a response_format is provided, try to process the content as JSON
    if response_format is not None:

        # Try to parse the response
        json_string = response
        from json_repair import repair_json

        repaired_json_string = repair_json(json_string)

        # If the repaired JSON is empty, return the original response
        if len(repaired_json_string.strip()) == 0 and len(json_string.strip()) > 0:
            return Message(role="assistant", text=response)

        # Try to validate the repaired JSON
        try:
            obj = response_format.model_validate_json(repaired_json_string)
            return Message(role="assistant", obj=obj)
        except Exception as e:
            return Message(role="assistant", text=response)
    else:
        # If no tool calls and no special format, return the plain message content
        return Message(role="assistant", text=response)
