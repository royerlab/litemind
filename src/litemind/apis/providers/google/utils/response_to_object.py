from typing import List, Optional

from pydantic import BaseModel

from litemind.agent.messages.message import Message


def response_to_object(
    messages: List[Message],
    model_name: str,
    max_num_output_tokens: int,
    response_format: Optional[BaseModel] = None,
) -> Message:
    """
    Generate a response from a message using the specified model.

    Parameters
    ----------
    messages: List[Message]
        Message object.
    model_name: str
        Model name.
    response_format: BaseModel
        Response format.

    Returns
    -------

    """

    import google.generativeai as genai
    from google.generativeai import types

    generation_cfg = types.GenerationConfig(
        temperature=0,
        max_output_tokens=max_num_output_tokens,
        response_mime_type="application/json",
        response_schema=response_format,
    )

    # Get model by name and set tools and config:
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_cfg,
    )

    # Convert messages to text:
    message_text = str(messages)

    # Generate content:
    gemini_response = model.generate_content(message_text)

    # Create a message object:
    message = Message(role="assistant", text=gemini_response.text)

    return message
