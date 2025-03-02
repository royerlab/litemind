from typing import Any, Dict, List

from litemind.agent.tools.toolset import ToolSet


def format_tools_for_openai(toolset: ToolSet) -> List[Dict[str, Any]]:
    """
    Convert ToolSet into JSON schema format required by OpenAI API.

    Parameters
    ----------
    toolset : ToolSet
        The ToolSet object to convert.

    Returns
    -------
    list of dict
        A list of dictionaries formatted for OpenAI's API.


    """
    tools = []
    for tool in toolset.list_tools():
        # Extract the properties and required fields from the JSON schema
        properties = tool.arguments_schema.get("properties", {})
        required = tool.arguments_schema.get("required", [])

        tool_schema = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }
        tools.append(tool_schema)
    return tools
