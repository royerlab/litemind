import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.agent.tools.agent_tool import AgentTool
from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.model_features import ModelFeatures


# Sample function to tests adding a FunctionTool
def sample_function(x: int) -> int:
    """Sample function that returns the square of a number."""
    return x * x


# Test for ToolSet initialization and add_tool
def test_toolset_initialization_and_add_tool():
    # Initialize an empty ToolSet
    toolset = ToolSet()
    assert toolset.list_tools() == [], "ToolSet should be initialized empty"

    # Add a FunctionTool directly
    func_tool = FunctionTool(sample_function, "Sample function that squares a number")
    toolset.add_tool(func_tool)
    assert (
        len(toolset.list_tools()) == 1
    ), "ToolSet should contain one tool after adding"
    assert (
        toolset.list_tools()[0] == func_tool
    ), "The added tool should be the FunctionTool instance"


# Test for adding a FunctionTool through add_function_tool
def test_toolset_add_function_tool():
    # Initialize ToolSet
    toolset = ToolSet()

    # Add FunctionTool via add_function_tool
    toolset.add_function_tool(sample_function, "Sample function that squares a number")

    # Retrieve and validate the FunctionTool
    added_tool = toolset.get_tool("sample_function")
    assert isinstance(
        added_tool, FunctionTool
    ), "The tool should be an instance of FunctionTool"
    assert (
        added_tool.description == "Sample function that squares a number"
    ), "The tool description should match the description provided"

    # Test that the FunctionTool executes correctly
    result = added_tool._execute(x=3)
    assert result == 9, "The FunctionTool should return the square of the input"


# Test for adding an AgentTool through add_agent_tool
@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_toolset_add_agent_tool(api_class):
    from litemind.agent.agent import Agent

    # Initialize OpenAIApi and Agent for the AgentTool
    api = api_class()  # Assumes API key is available in the environment

    # Make sure that the API supports text generation, or skip test:
    if not api.has_model_support_for(ModelFeatures.TextGeneration):
        # skip pytest:
        pytest.skip(
            f"Skipping test for {api_class.__name__} as no text model is available."
        )

    # Initialize Agent
    agent = Agent(api=api, name="agent_tool")

    # Initialize ToolSet and add an AgentTool
    toolset = ToolSet()
    toolset.add_agent_tool(agent, "Sample agent tool for testing")

    # Retrieve and validate the AgentTool
    added_tool = toolset.get_tool("agent_tool")
    assert isinstance(
        added_tool, AgentTool
    ), "The tool should be an instance of AgentTool"
    assert (
        added_tool.description
        == "Sample agent tool for testing."  # Note the period at the end
    ), "The tool description should match the description provided"

    # Execute the AgentTool with a sample prompt
    response = added_tool._execute(prompt="Translate 'Hello' to French.")
    assert (
        "bonjour" in response.lower()
    ), "The response should contain 'bonjour' as part of the translation"


# Test for retrieving a tool by name
def test_toolset_get_tool():
    # Initialize ToolSet and add a FunctionTool
    toolset = ToolSet()
    toolset.add_function_tool(sample_function, "Sample function that squares a number")

    # Retrieve tool by name and check if it matches the added tool
    retrieved_tool = toolset.get_tool("sample_function")
    assert retrieved_tool is not None, "get_tool should return the tool if it exists"
    assert (
        retrieved_tool.name == "sample_function"
    ), "The tool name should match the function name"
    assert isinstance(
        retrieved_tool, FunctionTool
    ), "The retrieved tool should be an instance of FunctionTool"

    # Check that retrieving a non-existent tool returns None
    non_existent_tool = toolset.get_tool("non_existent_tool")
    assert (
        non_existent_tool is None
    ), "get_tool should return None for a non-existent tool"


# Test for listing all tools in the ToolSet
def test_toolset_list_tools():
    # Initialize ToolSet and add multiple tools
    toolset = ToolSet()
    toolset.add_function_tool(sample_function, "Sample function that squares a number")

    # Verify that list_tools returns all tools in the ToolSet
    tools = toolset.list_tools()
    assert len(tools) == 1, "ToolSet should contain one tool"
    assert isinstance(
        tools[0], FunctionTool
    ), "The first tool in the list should be a FunctionTool"
    assert (
        tools[0].name == "sample_function"
    ), "The tool name should match 'sample_function'"
