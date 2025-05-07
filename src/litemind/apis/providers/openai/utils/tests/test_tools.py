import pytest

from litemind.agent.tools.toolset import ToolSet
from litemind.apis.providers.openai.utils.check_availability import check_openai_api_availability
from litemind.apis.providers.openai.utils.format_tools import format_tools_for_openai


def sample_tool_function(order_id: str) -> str:
    """Retrieve delivery date for a given order ID."""
    return "2024-11-15"  # Placeholder return value


def test_format_tools_for_openai():

    # Check if OpenAI API key is available:
    if not check_openai_api_availability():
        # Skip the test if the API key is not available:
        print("Skipping test_convert_text_only_message: OpenAI API key not available.")
        pytest.skip("OpenAI API key not available.")

    # Initialize ToolSet and add a sample tool
    toolset = ToolSet()
    toolset.add_function_tool(
        sample_tool_function, "Retrieve delivery date for a given order ID"
    )

    # Generate the OpenAI-compatible tool schema
    formatted_tools = format_tools_for_openai(toolset)

    # Expected JSON schema format for the OpenAI API
    expected_output = [
        {
            "type": "function",
            "function": {
                "name": "sample_tool_function",
                "description": "Retrieve delivery date for a given order ID",
                "parameters": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    ]

    # Check that formatted_tools matches expected_output
    assert (
        formatted_tools == expected_output
    ), "Formatted tools schema does not match the expected OpenAI format."
