import json
from typing import Optional, Any

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet


def _process_response(response: Any,
                      toolset: Optional[ToolSet]) -> Message:
    """
    Process OpenAI response, checking for function calls and executing them if needed.

    Parameters
    ----------
    response : Any
        The response from OpenAI API.
    toolset : Optional[ToolSet]
        The ToolSet object containing the tools to execute.

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

    # If no tool call, return the direct assistant message content
    return Message(role='assistant', text=choice.content or "")