from typing import Dict, List

from litemind.agent.tools.toolset import ToolSet


def format_tools_for_ollama(toolset: ToolSet) -> List[Dict]:
    """
    Convert a litemind ToolSet into Ollama's expected `tools` JSON schema.

    Each tool in the ToolSet must provide:
      - name: str
      - description: str
      - parameters: dict (JSON schema)
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
