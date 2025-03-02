from litemind.agent.tools.function_tool import FunctionTool


# Sample functions for testing
def sample_function_one(param1: int, param2: str) -> str:
    """Sample function that returns a formatted string from two integers."""
    return f"Received {param1} and {param2}"


def sample_function_no_params() -> str:
    """Function with no parameters."""
    return "No parameters here!"


def sample_function_with_defaults(param1: int, param2: str = "default") -> str:
    """Function with a default parameter."""
    return f"Received {param1} and {param2}"


# Test creation of Tool and its properties with JSON schema expectations
def test_tool_initialization():
    tool = FunctionTool(
        func=sample_function_one, description="Test function with two parameters"
    )

    # Expected JSON schema format for parameters
    expected_parameters = {
        "type": "object",
        "properties": {"param1": {"type": "number"}, "param2": {"type": "string"}},
        "required": ["param1", "param2"],
        "additionalProperties": False,
    }

    # Assert that the tool parameters match the expected JSON schema format
    assert (
        tool.arguments_schema == expected_parameters
    ), "Tool parameters schema does not match the expected format"


# Test creation of Tool and its properties with JSON schema expectations
def test_tool_initialization_automatic_description():
    tool = FunctionTool(func=sample_function_one)

    # Check that the description matches the function's docstring:
    assert (
        tool.description
        == "Sample function that returns a formatted string from two integers."
    ), "Description does not match function docstring"


# Test parameter schema generation for function with no parameters
def test_tool_with_no_parameters():
    tool = FunctionTool(
        func=sample_function_no_params, description="Function with no parameters"
    )

    # Expected JSON schema format for a function with no parameters
    expected_parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    # Assert that the parameters schema is empty as expected
    assert (
        tool.arguments_schema == expected_parameters
    ), "Expected empty parameters for function with no parameters"


# Test parameter schema generation with default values
def test_tool_with_default_parameter():
    tool = FunctionTool(
        func=sample_function_with_defaults,
        description="Function with default parameter",
    )

    # Expected JSON schema format with required and optional parameters
    expected_parameters = {
        "type": "object",
        "properties": {"param1": {"type": "number"}, "param2": {"type": "string"}},
        "required": ["param1"],
        "additionalProperties": False,
    }

    # Assert that the tool parameters match the expected format with default values handled correctly
    assert (
        tool.arguments_schema == expected_parameters
    ), "Expected param2 to be optional due to default value"


# Test execution of function with parameters
def test_tool_execution_with_parameters():
    tool = FunctionTool(
        func=sample_function_one, description="Test function with parameters"
    )

    # Execute the tool and check the result
    result = tool.execute(param1=42, param2="example")
    assert (
        result == "Received 42 and example"
    ), "Execution result mismatch for sample_function_one"


# Test execution of function with no parameters
def test_tool_execution_no_parameters():
    tool = FunctionTool(
        func=sample_function_no_params, description="Function with no parameters"
    )

    # Execute the tool and check the result
    result = tool.execute()
    assert (
        result == "No parameters here!"
    ), "Execution result mismatch for sample_function_no_params"


# Test execution of function with default parameters
def test_tool_execution_with_default_parameters():
    tool = FunctionTool(
        func=sample_function_with_defaults,
        description="Function with default parameter",
    )

    # Test execution with default parameter
    result_with_default = tool.execute(param1=100)
    assert (
        result_with_default == "Received 100 and default"
    ), "Default parameter not applied correctly"

    # Test execution with overridden default parameter
    result_with_override = tool.execute(param1=100, param2="custom")
    assert (
        result_with_override == "Received 100 and custom"
    ), "Parameter override not applied correctly"
