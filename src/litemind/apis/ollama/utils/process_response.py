import json
from typing import Optional

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet


def _process_response(response: dict,
                      toolset: Optional[ToolSet]) -> Message:
    """
    Process Ollama's response, checking for tool calls and executing them if needed.

    Parameters
    ----------
    response : dict
        The response from Ollama's `chat` call, e.g.:
        {
          'done': True,
          'message': {
            'content': "...",
            'tool_calls': [
                {
                    'function': {...},
                    'type': 'function_call',
                }
            ]
          }
        }
    toolset : Optional[ToolSet]
        The ToolSet object containing the tools to execute.

    Returns
    -------
    Message
        The final response message (either direct text or the tool's result).
    """
    # The top-level content from Ollama
    message_data = response["message"]
    text_content = message_data.get("content") or ""

    # 1) Check for tool calls
    tool_calls = message_data.get("tool_calls", [])
    if tool_calls and toolset:
        # For simplicity, just handle the first tool call
        tool_call = tool_calls[0]
        func_info = tool_call.get("function", {})
        function_name = func_info.get("name", "")
        arguments = func_info.get("arguments", {})

        # If arguments is a JSON-encoded string, parse it
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                # fallback if it's not valid JSON
                pass

        # 2) Execute the corresponding tool
        tool = toolset.get_tool(function_name) if function_name else None
        if tool:
            try:
                result = tool.execute(**arguments)
                # Return the tool’s result as JSON
                response_content = json.dumps(result, default=str)
                return Message(role='assistant', text=response_content)
            except Exception as e:
                err_msg = f"Function '{function_name}' error: {e}"
                return Message(role='assistant', text=err_msg)

    # 3) If no tool calls or no toolset, just return the text content
    return Message(role='assistant', text=text_content)
