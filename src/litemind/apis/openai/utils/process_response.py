import json
from typing import Optional, Any, Callable

from pydantic import BaseModel

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet


def process_response_from_openai(response: Any,
                                 toolset: Optional[ToolSet],
                                 response_format: Optional[BaseModel | str],
                                 stream_callback: Callable[[str, dict], None] = None) -> Message:
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
    stream_callback : Callable[[str, dict],None]
        The callback function to call when a stream message is received.

    Returns
    -------
    Message
        The response message to return.

    """

    #
    choice = response.choices[0].message

    # Check if there are any tool calls (function calls)
    if choice.tool_calls:
        results = []
        # Process each tool call in the list
        for tool_call in choice.tool_calls:
            function_name = tool_call.function.name
            try:
                arguments = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError as e:
                arguments = {}
            # Look up the tool using its name
            tool = toolset.get_tool(function_name) if toolset else None
            if tool:
                try:
                    result = tool.execute(**arguments)
                    results.append({"tool": function_name, "result": result})
                except Exception as e:
                    results.append({"tool": function_name, "error": str(e)})
            else:
                results.append({"tool": function_name, "error": "Tool not found"})
        # Aggregate all tool results into one JSON response
        aggregated_response = json.dumps(results, default=str)
        return Message(role='assistant', text=aggregated_response)

    # If a response_format is provided, try to process the content as JSON
    if response_format is not None:
        json_string = choice.content
        from json_repair import repair_json
        repaired_json_string = repair_json(json_string)
        if len(repaired_json_string.strip()) == 0 and len(json_string.strip()) > 0:
            return Message(role='assistant', text=choice.content)
        try:
            obj = response_format.model_validate_json(repaired_json_string)
            return Message(role='assistant', obj=obj)
        except Exception as e:
            return Message(role='assistant', text=choice.content)

    # If no tool calls and no special format, return the plain message content
    return Message(role='assistant', text=choice.content or "")
