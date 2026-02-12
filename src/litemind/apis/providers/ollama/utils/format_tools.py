"""Utility for formatting litemind tools into Ollama's tool schema.

Converts a litemind ``ToolSet`` into the JSON-based tool definition
format expected by the Ollama chat API for function-calling support.
"""

from typing import Dict, List

from litemind.agent.tools.toolset import ToolSet


def format_tools_for_ollama(toolset: ToolSet) -> List[Dict]:
    """Convert a litemind ToolSet into Ollama's tool JSON schema format.

    Parameters
    ----------
    toolset : ToolSet
        The toolset containing function tools to convert.

    Returns
    -------
    List[Dict]
        List of tool definitions in Ollama's expected format.
    """
    ollama_tools = []
    for tool in toolset.tools:  # Adjust as needed if your toolset differs
        ollama_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.arguments_schema,  # Must be a valid JSON schema
                },
            }
        )
    return ollama_tools
