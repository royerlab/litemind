import json
from typing import Optional, Any

from pydantic import BaseModel

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet


def process_response_from_openai(response: Any,
                                 toolset: Optional[ToolSet],
                                 response_format: Optional[BaseModel | str]) -> Message:
    """
    Process OpenAI response, checking for function calls and executing them if needed.

    Parameters
    ----------
    response : Any
        The response from OpenAI API.
    toolset : Optional[ToolSet]
        The ToolSet object containing the tools to execute.
    response_format : Optional[BaseModel | str]
        The format of the response.

    Returns
    -------
    Message
        The response message to return.

    """

    choice = response.choices[0].message

    # Check if there is a tool call (function call)
    if choice.tool_calls:
        tool_call = choice.tool_calls[0]
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)

        # Find and execute the tool function
        tool = toolset.get_tool(function_name) if toolset else None
        if tool:
            try:
                result = tool.execute(**arguments)
                response_content = json.dumps(result, default=str)
            except Exception as e:
                response_content = f"Function '{function_name}' error: {e}"

            return Message(role='assistant', text=response_content)

    if response_format is not None:

        # JSON string :
        json_string = choice.content

        # We first try to repair the JSON string:
        from json_repair import repair_json
        repaired_json_string = repair_json(json_string)

        # If the result is an empty string the string was not a JSON string or too broken:
        if len(repaired_json_string.strip()) == 0 and len(json_string.strip()) > 0:
            # In this case we just return the content of the message:
            return Message(role='assistant', text=choice.content)

        # We parse the Json string into the object:
        try:
            obj = response_format.model_validate_json(repaired_json_string)
            return Message(role='assistant', obj=obj)
        except Exception as e:
            # If parsing fails, return the content of the message:
            return Message(role='assistant', text=choice.content)

    # If no tool call, return the direct assistant message content
    return Message(role='assistant', text=choice.content or "")
