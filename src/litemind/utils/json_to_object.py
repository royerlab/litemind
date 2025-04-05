from typing import Optional, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message


def json_to_object(
    message: Message,
    response_format: Optional[Union[BaseModel, str]],
    text_content: str,
):
    """
    Parse a JSON string into an object of the specified format.

    Parameters
    ----------
    message: Message
        The message object to append the parsed JSON object to.
    response_format: Optional[BaseModel | str]
        The format of the JSON string to parse.
    text_content: str
        The JSON string to parse.

    Returns
    -------
    Message
        The message object with the parsed JSON object appended.

    """

    # Strip the text content of leading and trailing whitespace:
    text_content = text_content.strip()

    # Attempt to repair the JSON string
    from json_repair import repair_json

    repaired_json = repair_json(text_content)

    # Ensure that the repaired JSON string is stripped of leading and trailing whitespace:
    repaired_json = repaired_json.strip()

    # If the repaired string is empty, return the original text content
    if len(repaired_json) == 0 and len(text_content) > 0:
        # If the JSON string is empty, no need to do anything.
        pass
    else:
        try:
            # Parse the JSON string into the specified format
            parsed_obj = response_format.model_validate_json(repaired_json)
            message.append_object(parsed_obj)
        except Exception as e:
            # If an error occurs, append an error message
            message.append_text(f"Error parsing JSON: {e}")
    return message
