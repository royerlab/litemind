import json
from typing import Optional, Any

from pydantic import BaseModel

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet


def _process_response(response: Any,
                      toolset: Optional[ToolSet],
                      response_format: Optional[BaseModel | str] = None) -> Message:
    """
    Process Anthropic response, checking for function calls and executing them if needed.

    Parameters
    ----------
    response : Any
        The response from Anthropic API.
    toolset : Optional[ToolSet]
        The ToolSet object containing the tools to execute.
    response_format : Optional[BaseModel | str]
        The format of the response.

    Returns
    -------
    Message
        The response message to return.

    """
    # If the model wants to use a tool, parse out the tool call:
    if response.stop_reason == "tool_use":
        tool_item = next((c for c in response.content if c.type == "tool_use"),
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
                               text=f"(Tool '{function_name}' use requested, but tool not found.)")
        else:
            return Message(role="assistant",
                           text=f"(Tool '{tool_item}' use requested, but no details found.)")
    else:
        content = response.content
        if isinstance(content, str):
            text_content = content
        else:
            text_parts = [c.text for c in content if c.type == "text"]
            text_content = "".join(text_parts)

        if response_format:
            # Attempt to repair the JSON string
            from json_repair import repair_json
            repaired_json = repair_json(text_content)

            # If the repaired string is empty, return the original text content
            if len(repaired_json.strip()) == 0 and len(text_content.strip()) > 0:
                return Message(role='assistant', text=text_content)

            # Parse the JSON string into the specified format
            try:
                parsed_obj = response_format.model_validate_json(repaired_json)
                return Message(role='assistant', obj=parsed_obj)
            except Exception as e:
                # If parsing fails, return the original text content
                return Message(role='assistant', text=text_content)

        return Message(role="assistant", text=text_content)
