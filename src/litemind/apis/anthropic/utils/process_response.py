import json
from typing import Optional, Any

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet


def _process_response(response: Any, toolset: Optional[ToolSet]) -> Message:
    """
    Process Anthropic response, checking for function calls and executing them if needed.

    Parameters
    ----------
    response : Any
        The response from Anthropic API.
    toolset : Optional[ToolSet]
        The ToolSet object containing the tools to execute.

    Returns
    -------
    Message
        The response message to return.

    """
    # If the model wants to use a tool, parse out the tool call:
    if response.stop_reason == "tool_use":
        tool_item = next((c for c in response.content if c.type == "tool_use"),
                         None)
        text = next((c.text for c in response.content if c.type == "text"),
                    None)

        if tool_item:
            function_name = tool_item.name

            tool = toolset.get_tool(function_name) if toolset else None
            if tool:
                try:
                    result = tool.execute(**tool_item.input)
                    response_content = json.dumps(result, default=str)
                except Exception as e:
                    response_content = f"Function '{function_name}' error: {e}"

                return Message(role="assistant", text=response_content)
            else:
                return Message(role="assistant",
                               text="(Tool use requested, but tool not found.)")
        else:
            return Message(role="assistant",
                           text="(Tool use requested, but no details found.)")
    else:
        content = response.content
        if isinstance(content, str):
            text = content
        else:
            text_parts = [c.text for c in content if c.type == "text"]
            text = "".join(text_parts)

        return Message(role="assistant", text=text)
