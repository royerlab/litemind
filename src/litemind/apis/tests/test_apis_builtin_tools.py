import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.ressources.media_resources import MediaResources


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsBuiltinTools(MediaResources):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    These tests check builtin tools
    """

    def test_text_generation_with_builtin_web_search_tool(self, api_class):
        """
        Test a simple completion with the default or a known good model.
        """
        api_instance = api_class()

        # Get the best model for text generation:
        model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.WebSearchTool]
        )

        # If the model does not support text generation, skip the test:
        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and built-in web serach tool. Skipping tests."
            )

        # Print the model name:
        print("\n" + model_name)

        # A simple message:
        messages = [
            Message(
                role="system",
                text="You are a helpful assistant with access to a web search tool.",
            ),
            Message(
                role="user",
                text="In which 'classe preparatoire' did Loic A. Royer study?",
            ),
        ]

        # Create a ToolSet instance:
        toolset = ToolSet()
        # Add the built-in web search tool to the toolset:
        toolset.add_builtin_web_search_tool()

        # Get the generated text:
        response = api_instance.generate_text(
            model_name=model_name,
            messages=messages,
            temperature=0.0,
            toolset=toolset,  # Pass the toolset with the built-in web search tool
        )

        # There should be only one message in the response:
        assert len(response) == 1, f"Expected only one message in the response."

        # Extract the message from the list:
        response = response[0]

        # Print the response:
        print("\n" + str(response))

        assert (
            response.role == "assistant"
        ), f"{api_class.__name__} completion should return an 'assistant' role."

        # Extract the text from the response:
        response_text = str(response).lower()

        # Check that 'Janson' or 'Sailly' is in the response:
        assert any(
            "janson" in response_text
            or "sailly" in response_text
            or "dresden" in response_text
            or "robotics" in response_text
            for response in response
        ), f"{api_class.__name__} completion should mention 'Janson', 'Sailly', 'Dresden' or 'Robotics' in the response."

    def test_text_generation_with_builtin_mcp_tool(self, api_class):
        """
        Test a simple completion with the default or a known good model.
        """
        api_instance = api_class()

        # Get the best model for text generation:
        model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.MCPTool]
        )

        # If the model does not support text generation, skip the test:
        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation and built-in MCP tool. Skipping tests."
            )

        # Print the model name:
        print("\n" + model_name)

        # A simple message:
        messages = [
            Message(
                role="system",
                text="You are a helpful assistant.",
            ),
            Message(
                role="user",
                text="What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
            ),
        ]

        # Create a ToolSet instance:
        toolset = ToolSet()
        # Add the built-in web search tool to the toolset:
        toolset.add_builtin_mcp_tool(
            server_name="deepwiki",
            server_url="https://mcp.deepwiki.com/mcp",
            allowed_tools=["ask_question"],
        )

        # Get the generated text:
        response = api_instance.generate_text(
            model_name=model_name,
            messages=messages,
            temperature=0.0,
            toolset=toolset,  # Pass the toolset with the built-in web search tool
        )

        # There should be only one message in the response:
        assert len(response) == 1, f"Expected only one message in the response."

        # Extract the message from the list:
        response = response[0]

        # Print the response:
        print("\n" + str(response))

        assert (
            response.role == "assistant"
        ), f"{api_class.__name__} completion should return an 'assistant' role."

        # Extract the text from the response:
        response_text = str(response).lower()

        # Check that 'stdio' or 'Streamable HTTP Transport' is in the response:
        assert any(
            "stdio" in response_text or "streamable http transport" in response_text
            for response in response
        ), f"{api_class.__name__} completion should mention 'stdio' or 'Streamable HTTP Transport' in the response."
