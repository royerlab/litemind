"""Utility for processing Ollama chat responses into litemind messages.

Converts an Ollama ``ChatResponse`` (or stream thereof) into a litemind
``Message`` object, extracting text content, thinking blocks (delimited
by ``<thinking>`` tags), tool calls, and optional structured output
parsed from JSON.
"""

from typing import Iterator, Optional, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.utils.extract_thinking import extract_thinking_content
from litemind.utils.json_to_object import json_to_object


def process_response_from_ollama(
    ollama_response: Union["ChatResponse", Iterator["ChatResponse"]],
    toolset: Optional[ToolSet],
    response_format: Optional[Union[BaseModel, str]] = None,
) -> Message:
    """Convert an Ollama chat response into a litemind Message.

    Extracts text content, thinking blocks, and tool calls from the
    Ollama response. Optionally parses structured output.

    Parameters
    ----------
    ollama_response : Union[ChatResponse, Iterator[ChatResponse]]
        The response from Ollama's ``chat`` call.
    toolset : Optional[ToolSet]
        The ToolSet used during generation, needed for tool call extraction.
    response_format : Optional[Union[BaseModel, str]]
        If provided, parses the text response into this Pydantic model.

    Returns
    -------
    Message
        Processed response as a litemind Message with role "assistant".
    """

    # Initialize the processed response:
    processed_response = Message(role="assistant")

    # Text content:
    text_content = ""

    # Variable to hold whether the response is a tool use:
    is_tool_use = False

    # The top-level content from Ollama
    ollama_message = ollama_response["message"]

    # Get text message:
    text_message = ollama_message.get("content", "")

    # Extract thinking block from Ollama message:
    text_message, thinking = extract_thinking_content(text_message)

    if thinking is not None and len(thinking) > 0:
        processed_response.append_thinking(thinking)

    # Append the text message to the processed response:
    if text_message.strip():
        processed_response.append_text(text_message)

    # Append the text message to the text content:
    text_content += text_message

    # Get the tool calls from the Ollama message:
    tool_calls = ollama_message.get("tool_calls", [])

    if tool_calls and toolset:
        is_tool_use = True
        for tool_call in tool_calls:
            func_info = tool_call.get("function", {})
            function_name = func_info.get("name", "")
            arguments = func_info.get("arguments", {})

            processed_response.append_tool_call(
                tool_name=function_name, arguments=arguments, id=""
            )

    # If response_format is provided, try to parse the text content
    if response_format and not is_tool_use:
        processed_response = json_to_object(
            processed_response, response_format, text_content
        )

    # If no tool calls or no toolset, just return the text content
    return processed_response
