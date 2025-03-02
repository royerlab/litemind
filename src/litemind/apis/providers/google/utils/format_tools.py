from typing import List


def format_tools_for_gemini(toolset) -> List["Tool"]:
    """
    Convert your custom tool objects into genai.protos.Tool
    for fine-grained control of parameter schemas and descriptions.

    Each BaseTool is assumed to have:
        - name (string)
        - description (string)
        - parameters (JSON schema)  # from FunctionTool._generate_parameters_schema()
    """

    from google.generativeai import protos

    # Check if the toolset is None, if yes return None:
    if not toolset:
        return None

    # Initialize an empty list of tools:
    tools = []

    # Iterate over the tools in the toolset:
    for tool in toolset.list_tools():

        # Check if they are any parameters:
        if tool.arguments_schema["properties"]:
            # Create a FunctionDeclaration for each tool:
            func_decl = protos.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "No description",
                parameters=_create_protos_schema(tool.arguments_schema),
            )
        else:
            # Create a FunctionDeclaration for each tool:
            func_decl = protos.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "No description",
            )

        # Wrap it in a Tool
        tool_proto = protos.Tool(function_declarations=[func_decl])
        tools.append(tool_proto)

    return tools


def _create_protos_schema(json_schema: dict) -> "Schema":
    """
    Convert a JSON-schema-like dict (such as tool.parameters) into a genai.protos.Schema.
    This supports a single level of object properties: "type": "object", "properties": {...}.
    Extend as needed for nested or more complex structures.
    """
    from google.generativeai import protos

    # Map JSON-schema type strings to Gemini (proto) types
    type_map = {
        "string": protos.Type.STRING,
        "number": protos.Type.NUMBER,
        "boolean": protos.Type.BOOLEAN,
        "array": protos.Type.ARRAY,
        "object": protos.Type.OBJECT,
    }

    # Root-level type
    root_type_str = json_schema.get("type", None)

    root_type = type_map.get(root_type_str, protos.Type.STRING)

    schema = protos.Schema(type_=root_type)

    if root_type == protos.Type.OBJECT:
        # For each property in the schema, set up a child protos.Schema
        for prop_name, prop_def in json_schema.get("properties", {}).items():
            prop_type_str = prop_def.get("type", "string")
            prop_type = type_map.get(prop_type_str, protos.Type.STRING)
            child_schema = protos.Schema(type_=prop_type)
            schema.properties[prop_name] = child_schema

        # Mark required fields
        required_fields = json_schema.get("required", [])
        for field_name in required_fields:
            schema.required.append(field_name)

    return schema
