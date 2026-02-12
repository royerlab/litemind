"""Conversion of litemind ToolSet to Gemini tool and schema format."""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from google.genai.types import Schema, Tool


def format_tools_for_gemini(toolset) -> Optional[List["Tool"]]:
    """Convert a ToolSet into Gemini's Tool format with FunctionDeclarations.

    Parameters
    ----------
    toolset : Optional[ToolSet]
        The toolset to convert. If None or empty, returns None.

    Returns
    -------
    Optional[List[Tool]]
        List of google.genai Tool objects, or None if no tools.
    """

    from google.genai import types

    # Check if the toolset is None, if yes return None:
    if not toolset:
        return None

    # Collect all function declarations into a single Tool object.
    # Gemini expects all declarations in one Tool, not one Tool per function.
    func_decls = []

    # Iterate over the tools in the toolset:
    for tool in toolset.list_tools():

        # Check if there are any parameters (defensive check for "properties" key):
        has_properties = (
            tool.arguments_schema
            and "properties" in tool.arguments_schema
            and tool.arguments_schema["properties"]
        )
        if has_properties:
            # Create a FunctionDeclaration with parameter schema:
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "No description",
                parameters=_create_schema(tool.arguments_schema),
            )
        else:
            # Gemini requires an explicit empty parameters schema for
            # no-argument functions, otherwise it returns MALFORMED_FUNCTION_CALL.
            func_decl = types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "No description",
                parameters=types.Schema(type="OBJECT", properties={}),
            )

        func_decls.append(func_decl)

    if not func_decls:
        return None

    return [types.Tool(function_declarations=func_decls)]


def _create_schema(json_schema: dict) -> "Schema":
    """Convert a JSON Schema dict into a google.genai Schema object.

    Supports a single level of object properties. Nested or complex
    structures may need extension.

    Parameters
    ----------
    json_schema : dict
        JSON Schema dict with ``type``, ``properties``, and ``required`` keys.

    Returns
    -------
    Schema
        A google.genai Schema instance.
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
