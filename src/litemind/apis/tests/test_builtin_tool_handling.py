"""
Tests for built-in tool handling across all providers.

These tests verify:
- Built-in tools don't break custom tool usage across all providers
- Web search + custom tools can coexist in the same request
- Multi-turn conversations with web search work (round-trip serialization)

All tests are parametrized across available API implementations.
"""

import pytest

from litemind import get_available_apis
from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.ressources.media_resources import MediaResources


@pytest.mark.parametrize("api_class", get_available_apis())
class TestBuiltinToolHandling(MediaResources):
    """
    Tests that built-in tool handling works correctly across all providers.
    Covers format_tools filtering, tool call processing, and round-trip
    serialization for web search and MCP tools.
    """

    def test_builtin_tools_do_not_break_custom_tools(self, api_class):
        """
        Adding a BuiltinWebSearchTool to a toolset should not break custom
        tool usage, even for providers that don't support web search natively.
        The builtin should be silently filtered, and custom tools should work.
        """
        api_instance = api_class()

        # Get a model that supports text generation and custom tools:
        model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Tools]
        )

        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support text generation with tools. Skipping."
            )

        print("\n" + model_name)

        # Define a simple custom tool:
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            return f"Sunny, 25°C in {city}"

        # Create toolset with BOTH a builtin web search tool and a custom tool:
        toolset = ToolSet()
        toolset.add_builtin_web_search_tool()
        toolset.add_function_tool(get_weather)

        messages = [
            Message(
                role="system",
                text="You are a helpful assistant. Use the get_weather tool to answer weather questions.",
            ),
            Message(
                role="user",
                text="What is the weather in Tokyo? Use the get_weather tool.",
            ),
        ]

        # This should not crash regardless of whether the provider supports web search:
        response = api_instance.generate_text(
            model_name=model_name,
            messages=messages,
            temperature=0.0,
            toolset=toolset,
        )

        # Should get a response:
        assert response is not None
        assert len(response) >= 1

        # Print the response:
        print("\n" + str(response[-1]))

        # The response should mention Tokyo (from the custom tool result):
        response_text = str(response[-1]).lower()
        assert (
            "tokyo" in response_text
        ), f"{api_class.__name__} response should mention 'Tokyo'."

    def test_web_search_with_custom_tools_combined(self, api_class):
        """
        When a provider supports web search, using both web search and custom
        tools in the same request should work without errors. The model should
        be able to use web search AND call custom tools in the same turn.
        """
        api_instance = api_class()

        # Need a model that supports both web search and custom tools:
        model_name = api_instance.get_best_model(
            [
                ModelFeatures.TextGeneration,
                ModelFeatures.WebSearchTool,
                ModelFeatures.Tools,
            ]
        )

        if model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support web search + tools. Skipping."
            )

        print("\n" + model_name)

        # Define a formatting tool:
        def format_answer(text: str) -> str:
            """Format the answer text in uppercase."""
            return text.upper()

        toolset = ToolSet()
        toolset.add_builtin_web_search_tool()
        toolset.add_function_tool(format_answer)

        messages = [
            Message(
                role="system",
                text="You are a helpful assistant. After searching the web, always use the format_answer tool to format your final answer.",
            ),
            Message(
                role="user",
                text="What year was the Eiffel Tower built? Search the web, then use format_answer with the answer.",
            ),
        ]

        # Should handle mixed built-in + custom tools without crashing:
        response = api_instance.generate_text(
            model_name=model_name,
            messages=messages,
            temperature=0.0,
            toolset=toolset,
        )

        assert response is not None
        assert len(response) >= 1
        print("\n" + str(response[-1]))

    def test_web_search_multi_turn_round_trip(self, api_class):
        """
        Multi-turn conversations with web search should work correctly.
        The server_tool_use and web_search_tool_result blocks from the first
        turn must be properly round-tripped to the API in the second turn.
        This tests the fix for the BadRequestError on round-trip (Issue #10).
        """
        api_instance = api_class()

        model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.WebSearchTool]
        )

        if model_name is None:
            pytest.skip(f"{api_class.__name__} does not support web search. Skipping.")

        print("\n" + model_name)

        toolset = ToolSet()
        toolset.add_builtin_web_search_tool()

        # First turn: trigger web search
        messages = [
            Message(
                role="system",
                text="You are a helpful assistant with web search access.",
            ),
            Message(
                role="user",
                text="What is the population of Iceland? Search the web.",
            ),
        ]

        first_response = api_instance.generate_text(
            model_name=model_name,
            messages=messages,
            temperature=0.0,
            toolset=toolset,
        )

        assert first_response is not None
        assert len(first_response) >= 1
        print("\nFirst turn: " + str(first_response[-1]))

        # Second turn: follow-up question using the same conversation.
        # Note: generate_text already appended the response to messages,
        # so we only need to add the new user message.
        messages.append(
            Message(
                role="user",
                text="And what is the capital of that country?",
            ),
        )

        # This used to crash with BadRequestError due to non-string
        # web_search_tool_result and incorrect block type serialization:
        second_response = api_instance.generate_text(
            model_name=model_name,
            messages=messages,
            temperature=0.0,
            toolset=toolset,
        )

        assert second_response is not None
        assert len(second_response) >= 1
        print("\nSecond turn: " + str(second_response[-1]))

        # The response should mention Reykjavik:
        response_text = str(second_response[-1]).lower()
        assert (
            "reykjavik" in response_text or "reykjavík" in response_text
        ), f"{api_class.__name__} second turn should mention 'Reykjavik'."
