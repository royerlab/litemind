from typing import Optional, List

from anthropic.types import ToolParam

from litemind.agent.tools.toolset import ToolSet


def _convert_toolset_to_anthropic(toolset: Optional[ToolSet]) -> Optional[
    List[ToolParam]]:
    """
    Convert a ToolSet into Anthropic's list[ToolParam] format.

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

    tools: List[ToolParam] = []
    for tool in toolset.list_tools():
        tool_def = {
            "name": tool.name,
            "description": tool.description,
            # If your tool has a JSON schema, attach it:
            "input_schema": getattr(tool, "input_schema",
                                    {"type": "object"}),
        }
        tools.append(tool_def)
    return tools
