from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from google.genai.types import Schema, Tool


def format_tools_for_gemini(toolset) -> Optional[List["Tool"]]:
    """
    Convert your custom tool objects into google.genai types.Tool
    for fine-grained control of parameter schemas and descriptions.

    Each BaseTool is assumed to have:
        - name (string)
        - description (string)
        - parameters (JSON schema)  # from FunctionTool._generate_parameters_schema()
    """

    from google.genai import types

    # Check if the toolset is None, if yes return None:
    if not toolset:
        return None

    # Initialize an empty list of tools:
    tools = []

    # Iterate over the tools in the toolset:
    for tool in toolset.list_tools():

        # Check if there are any parameters (defensive check for "properties" key):
        has_properties = (
            tool.arguments_schema
            and "properties" in tool.arguments_schema
            and tool.arguments_schema["properties"]
        )
        if has_properties:
            # Create a FunctionDeclaration for each tool:
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "No description",
                parameters=_create_schema(tool.arguments_schema),
            )
        else:
            # Create a FunctionDeclaration for each tool:
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "No description",
            )

        # Wrap it in a Tool
        tool_obj = types.Tool(function_declarations=[func_decl])
        tools.append(tool_obj)

    return tools


def _create_schema(json_schema: dict) -> "Schema":
    """
    Convert a JSON-schema-like dict (such as tool.parameters) into a google.genai types.Schema.
    This supports a single level of object properties: "type": "object", "properties": {...}.
    Extend as needed for nested or more complex structures.
    """
    from google.genai import types

    # Map JSON-schema type strings to Gemini (types) types
    type_map = {
        "string": "STRING",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
        "integer": "INTEGER",
    }

    # Root-level type
    root_type_str = json_schema.get("type", None)
    root_type = type_map.get(root_type_str, "STRING")

    # Build properties dict for object types
    properties = None
    required = None

    if root_type == "OBJECT":
        properties = {}
        # For each property in the schema, set up a child Schema
        for prop_name, prop_def in json_schema.get("properties", {}).items():
            prop_type_str = prop_def.get("type", "string")
            prop_type = type_map.get(prop_type_str, "STRING")
            properties[prop_name] = types.Schema(type=prop_type)

        # Mark required fields
        required_fields = json_schema.get("required", [])
        if required_fields:
            required = required_fields

    return types.Schema(type=root_type, properties=properties, required=required)
