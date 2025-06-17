from typing import List, Optional

from litemind.agent.tools.toolset import ToolSet


def format_tools_for_anthropic(
    toolset: Optional[ToolSet],
) -> Optional[List["ToolParam"]]:
    """
    Convert a ToolSet into Anthropic's list[ToolParam] format for custom function tools.

    Built-in tools (web search, MCP, code execution) are handled separately
    by the main API class and should not be processed here.

    Example ToolParam:
    {
        "name": "get_weather",
        "description": "Get the weather for a specific location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {"type": "string"}
            }
        }
    }
    """
    if not toolset:
        return None

    from anthropic.types import ToolParam

    # Convert each custom tool in the ToolSet to an Anthropic ToolParam
    # Skip built-in tools as they are handled separately
    tools: List[ToolParam] = []
    for tool in toolset.list_tools():
        # Skip built-in tools - they are handled by the main API class
        if tool.is_builtin():
            continue

        tool_def = {
            "name": tool.name,
            "description": tool.description,
            # If your tool has a JSON schema, attach it:
            "input_schema": tool.arguments_schema,
        }
        tools.append(tool_def)

    return tools
