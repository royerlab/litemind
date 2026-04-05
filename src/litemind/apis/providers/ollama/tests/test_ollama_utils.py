"""Unit tests for the Ollama adapter utility functions.

Tests cover the changes made to:
- convert_messages_for_ollama (tool_name, tool call format, no response_format injection)
- process_response_from_ollama (native thinking field with fallback)
- aggregate_chat_responses (thinking field accumulation)
- OllamaApi._registry_supports_feature (prefix-based matching)
- OllamaApi._has_native_thinking (capabilities-based check)
"""

from unittest.mock import MagicMock, patch

import pytest

from litemind.agent.messages.actions.tool_use import ToolUse
from litemind.agent.messages.message import Message
from litemind.apis.providers.ollama.utils.convert_messages import (
    convert_messages_for_ollama,
)
from litemind.apis.providers.ollama.utils.process_response import (
    process_response_from_ollama,
)
from litemind.media.types.media_text import Text

# =========================================================================
# convert_messages_for_ollama tests
# =========================================================================


class TestConvertMessagesForOllama:
    """Tests for convert_messages_for_ollama."""

    def test_basic_text_message(self):
        """A simple user text message should convert correctly."""
        msg = Message(role="user", text="Hello world")
        result = convert_messages_for_ollama([msg])
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert "Hello world" in result[0]["content"]

    def test_thinking_blocks_are_skipped(self):
        """Thinking blocks should not appear in the output messages."""
        msg = Message(role="assistant")
        msg.append_thinking("internal reasoning")
        msg.append_text("final answer")
        result = convert_messages_for_ollama([msg])
        # Only the text block should produce a message, not the thinking block:
        assert len(result) == 1
        assert "final answer" in result[0]["content"]
        assert "internal reasoning" not in result[0]["content"]

    def test_tool_use_includes_tool_name(self):
        """Tool result messages must include tool_name (ollama >= 0.5.2)."""
        msg = Message(role="user")
        msg.append_tool_use(
            tool_name="get_weather",
            arguments={"city": "Paris"},
            result="Sunny, 22C",
            id="call_123",
        )
        result = convert_messages_for_ollama([msg])

        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_name"] == "get_weather"
        assert "Sunny, 22C" in result[0]["content"]

    def test_tool_call_has_no_spurious_type_field(self):
        """Tool call dicts should not have a 'type' key (was 'function_call' bug)."""
        msg = Message(role="assistant")
        msg.append_tool_call(tool_name="add", arguments={"a": 1, "b": 2}, id="c1")
        result = convert_messages_for_ollama([msg])

        assert len(result) == 1
        tool_calls = result[0]["tool_calls"]
        assert len(tool_calls) == 1
        tc = tool_calls[0]
        # Must NOT have a 'type' key:
        assert "type" not in tc
        # Must have correct function info:
        assert tc["function"]["name"] == "add"
        assert tc["function"]["arguments"] == {"a": 1, "b": 2}

    def test_no_response_format_injection(self):
        """The function should not append a response_format user message."""
        msg = Message(role="user", text="Tell me about Tokyo")
        result = convert_messages_for_ollama([msg])
        # Should be exactly 1 message, no extra format message:
        assert len(result) == 1
        assert "schema" not in result[0]["content"].lower()

    def test_empty_text_is_skipped(self):
        """Messages with only whitespace text should not produce output."""
        msg = Message(role="user", text="   ")
        result = convert_messages_for_ollama([msg])
        assert len(result) == 0


# =========================================================================
# process_response_from_ollama tests
# =========================================================================


class TestProcessResponseFromOllama:
    """Tests for process_response_from_ollama."""

    def _make_response(self, content="", thinking=None, tool_calls=None):
        """Helper to create a mock Ollama ChatResponse-like dict."""
        message = {"role": "assistant", "content": content}
        if thinking is not None:
            message["thinking"] = thinking
        if tool_calls is not None:
            message["tool_calls"] = tool_calls
        return {"message": message}

    def test_native_thinking_field_is_used(self):
        """When native thinking field is present, use it instead of parsing tags."""
        response = self._make_response(
            content="The answer is 42.",
            thinking="Let me think step by step...",
        )
        result = process_response_from_ollama(response, toolset=None)

        # Should have thinking block:
        has_thinking = any(b.is_thinking() for b in result.blocks)
        assert has_thinking
        # Thinking content should be the native field value:
        thinking_blocks = [b for b in result.blocks if b.is_thinking()]
        assert "Let me think step by step" in thinking_blocks[0].get_content()
        # Text should NOT have thinking tags stripped (wasn't there):
        text_blocks = [
            b for b in result.blocks if b.has_type(Text) and not b.is_thinking()
        ]
        assert len(text_blocks) == 1
        assert "The answer is 42" in text_blocks[0].get_content()

    def test_fallback_to_xml_thinking_tags(self):
        """When no native thinking field, parse <thinking> tags from content."""
        response = self._make_response(
            content="<thinking>reasoning here</thinking>The answer is 42.",
        )
        result = process_response_from_ollama(response, toolset=None)

        has_thinking = any(b.is_thinking() for b in result.blocks)
        assert has_thinking
        thinking_blocks = [b for b in result.blocks if b.is_thinking()]
        assert "reasoning here" in thinking_blocks[0].get_content()

    def test_fallback_to_think_tags(self):
        """Parse <think> tags (used by QWQ, DeepSeek) from content."""
        response = self._make_response(
            content="<think>step by step reasoning</think>The answer is 255.",
        )
        result = process_response_from_ollama(response, toolset=None)

        has_thinking = any(b.is_thinking() for b in result.blocks)
        assert has_thinking

    def test_no_thinking(self):
        """When no thinking present, no thinking blocks should be created."""
        response = self._make_response(content="Just a plain answer.")
        result = process_response_from_ollama(response, toolset=None)

        has_thinking = any(b.is_thinking() for b in result.blocks)
        assert not has_thinking
        text_blocks = [b for b in result.blocks if b.has_type(Text)]
        assert len(text_blocks) == 1

    def test_native_thinking_takes_priority_over_tags(self):
        """If both native thinking and XML tags exist, native should win."""
        response = self._make_response(
            content="<thinking>tag content</thinking>Text here.",
            thinking="Native thinking content",
        )
        result = process_response_from_ollama(response, toolset=None)

        thinking_blocks = [b for b in result.blocks if b.is_thinking()]
        assert len(thinking_blocks) == 1
        # Should contain the native thinking, not the tag content:
        assert "Native thinking content" in thinking_blocks[0].get_content()

    def test_tool_calls_extraction(self):
        """Tool calls from the response should be extracted correctly."""
        tool_calls = [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ]
        response = self._make_response(content="", tool_calls=tool_calls)

        # Create a mock toolset:
        mock_toolset = MagicMock()

        result = process_response_from_ollama(response, toolset=mock_toolset)
        # Should have an Action block:
        from litemind.media.types.media_action import Action

        action_blocks = [b for b in result.blocks if b.has_type(Action)]
        assert len(action_blocks) == 1

    def test_empty_thinking_not_added(self):
        """Empty native thinking field should not create a thinking block."""
        response = self._make_response(content="Answer.", thinking="")
        result = process_response_from_ollama(response, toolset=None)
        has_thinking = any(b.is_thinking() for b in result.blocks)
        assert not has_thinking


# =========================================================================
# aggregate_chat_responses tests
# =========================================================================


class TestAggregateChatResponses:
    """Tests for aggregate_chat_responses thinking accumulation."""

    def test_thinking_accumulation(self):
        """Thinking content should be accumulated across streaming chunks."""
        from ollama import ChatResponse
        from ollama import Message as OllamaMessage

        chunks = [
            ChatResponse(
                model="test",
                message=OllamaMessage(
                    role="assistant", content="", thinking="Step 1. "
                ),
                done=False,
            ),
            ChatResponse(
                model="test",
                message=OllamaMessage(
                    role="assistant", content="", thinking="Step 2. "
                ),
                done=False,
            ),
            ChatResponse(
                model="test",
                message=OllamaMessage(
                    role="assistant", content="Final answer.", thinking=""
                ),
                done=True,
            ),
        ]

        from litemind.apis.providers.ollama.utils.aggregate_chat_responses import (
            aggregate_chat_responses,
        )

        callback = MagicMock()
        result = aggregate_chat_responses(chunks, callback=callback)

        assert result.message.thinking == "Step 1. Step 2. "
        assert result.message.content == "Final answer."

    def test_content_accumulation(self):
        """Text content should still accumulate correctly."""
        from ollama import ChatResponse
        from ollama import Message as OllamaMessage

        chunks = [
            ChatResponse(
                model="test",
                message=OllamaMessage(role="assistant", content="Hello "),
                done=False,
            ),
            ChatResponse(
                model="test",
                message=OllamaMessage(role="assistant", content="world!"),
                done=True,
            ),
        ]

        from litemind.apis.providers.ollama.utils.aggregate_chat_responses import (
            aggregate_chat_responses,
        )

        callback = MagicMock()
        result = aggregate_chat_responses(chunks, callback=callback)

        assert result.message.content == "Hello world!"
        assert callback.call_count == 2

    def test_empty_chunks(self):
        """Empty chunk iterable should return a default ChatResponse."""
        from litemind.apis.providers.ollama.utils.aggregate_chat_responses import (
            aggregate_chat_responses,
        )

        callback = MagicMock()
        result = aggregate_chat_responses(iter([]), callback=callback)
        assert result.message.role == "assistant"
        assert result.message.content == ""


# =========================================================================
# OllamaApi._registry_supports_feature tests
# =========================================================================


class TestRegistrySupportsFeature:
    """Tests for OllamaApi._registry_supports_feature prefix matching."""

    @pytest.fixture
    def api(self):
        """Create an OllamaApi with mocked client to avoid needing a server."""
        with patch(
            "litemind.apis.providers.ollama.ollama_api.OllamaApi.__init__",
            lambda self, **kw: None,
        ):
            from litemind.apis.providers.ollama.ollama_api import OllamaApi

            api = OllamaApi.__new__(OllamaApi)
            # Initialize required attributes manually:
            from litemind.apis.model_registry import get_default_model_registry

            api.model_registry = get_default_model_registry()
            return api

    def test_exact_match(self, api):
        """Exact model name should match directly."""
        from litemind.apis.base_api import ModelFeatures

        # qwen2.5:7b is registered exactly in the YAML:
        assert api._registry_supports_feature("qwen2.5:7b", ModelFeatures.Tools) is True

    def test_base_name_fallback(self, api):
        """Model with unregistered tag should fall back to base name."""
        from litemind.apis.base_api import ModelFeatures

        # "qwen2.5:3b-q4_K_M" is NOT registered, but "qwen2.5" is:
        assert (
            api._registry_supports_feature("qwen2.5:3b-q4_K_M", ModelFeatures.Tools)
            is True
        )

    def test_quantization_stripping(self, api):
        """Quantization suffix should be stripped to find a registered entry."""
        from litemind.apis.base_api import ModelFeatures

        # "deepseek-r1:32b-q4_K_M" should match "deepseek-r1:32b" or "deepseek-r1":
        assert (
            api._registry_supports_feature(
                "deepseek-r1:32b-q4_K_M", ModelFeatures.TextGeneration
            )
            is True
        )

    def test_no_match_returns_false(self, api):
        """Unknown model should return False."""
        from litemind.apis.base_api import ModelFeatures

        assert (
            api._registry_supports_feature(
                "totally-unknown:model", ModelFeatures.TextGeneration
            )
            is False
        )

    def test_thinking_suffix_preserved_during_quantization_strip(self, api):
        """The -thinking suffix should be preserved when stripping quantization."""
        from litemind.apis.base_api import ModelFeatures

        # "deepseek-r1:32b-q4_K_M-thinking": after removing "-thinking" suffix,
        # tag becomes "32b-q4_K_M", parts=["32b","q4_K_M"], last starts with "q",
        # so simplified becomes "32b" + "-thinking" = "32b-thinking",
        # giving "deepseek-r1:32b-thinking" which IS registered with Thinking=true.
        result = api._registry_supports_feature(
            "deepseek-r1:32b-q4_K_M-thinking", ModelFeatures.Thinking
        )
        assert result is True

        # But the non-thinking quantized variant should NOT have Thinking:
        result = api._registry_supports_feature(
            "deepseek-r1:32b-q4_K_M", ModelFeatures.Thinking
        )
        assert result is False

    def test_vision_feature_on_gemma(self, api):
        """Gemma3 should have Image but not Tools."""
        from litemind.apis.base_api import ModelFeatures

        assert api._registry_supports_feature("gemma3:27b", ModelFeatures.Image) is True
        assert (
            api._registry_supports_feature("gemma3:27b", ModelFeatures.Tools) is False
        )

    def test_gemma3_1b_no_vision(self, api):
        """Gemma3 1b is text-only, no vision."""
        from litemind.apis.base_api import ModelFeatures

        assert api._registry_supports_feature("gemma3:1b", ModelFeatures.Image) is False
        assert (
            api._registry_supports_feature("gemma3:1b", ModelFeatures.TextGeneration)
            is True
        )

    def test_model_without_colon(self, api):
        """Models without ':' should use exact match only (no prefix stripping)."""
        from litemind.apis.base_api import ModelFeatures

        # "mistral" is registered as a base name with mistral-family (has Tools):
        assert api._registry_supports_feature("mistral", ModelFeatures.Tools) is True
        assert (
            api._registry_supports_feature("mistral", ModelFeatures.TextGeneration)
            is True
        )
        # Unregistered name without colon has no fallback path:
        assert (
            api._registry_supports_feature("nonexistent", ModelFeatures.TextGeneration)
            is False
        )

    def test_thinking_variants_never_report_embeddings(self, api):
        """Synthetic -thinking models must not report TextEmbeddings support.

        The -thinking suffix is only stripped in generate_text, not embed_texts.
        If a -thinking variant reported TextEmbeddings, it would be selected for
        embeddings and fail because the model name is invalid for that API.
        """
        from litemind.apis.base_api import ModelFeatures

        # Unregistered -thinking variant of a family WITH TextEmbeddings:
        assert (
            api._registry_supports_feature(
                "mistral:7b-thinking", ModelFeatures.TextEmbeddings
            )
            is False
        )
        assert (
            api._registry_supports_feature(
                "command-r:latest-thinking", ModelFeatures.TextEmbeddings
            )
            is False
        )
        # But TextGeneration should still work via fallback:
        assert (
            api._registry_supports_feature(
                "mistral:7b-thinking", ModelFeatures.TextGeneration
            )
            is True
        )
        # Registered -thinking variant also reports False (explicit override):
        assert (
            api._registry_supports_feature(
                "deepseek-r1:32b-thinking", ModelFeatures.TextEmbeddings
            )
            is False
        )


# =========================================================================
# process_response_from_ollama edge case tests
# =========================================================================


class TestProcessResponseEdgeCases:
    """Edge case tests for process_response_from_ollama."""

    def test_none_content_does_not_crash(self):
        """Response with content=None should not crash."""
        response = {"message": {"role": "assistant", "content": None}}
        result = process_response_from_ollama(response, toolset=None)
        # Should not crash and should have no text blocks:
        text_blocks = [
            b for b in result.blocks if b.has_type(Text) and not b.is_thinking()
        ]
        assert len(text_blocks) == 0

    def test_none_content_with_native_thinking(self):
        """Response with content=None but native thinking should work."""
        response = {
            "message": {
                "role": "assistant",
                "content": None,
                "thinking": "I am thinking...",
            }
        }
        result = process_response_from_ollama(response, toolset=None)
        thinking_blocks = [b for b in result.blocks if b.is_thinking()]
        assert len(thinking_blocks) == 1
        assert "I am thinking" in thinking_blocks[0].get_content()

    def test_none_content_with_tool_calls(self):
        """Response with content=None and tool_calls should work."""
        response = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "add", "arguments": {"a": 1, "b": 2}}}
                ],
            }
        }
        mock_toolset = MagicMock()
        result = process_response_from_ollama(response, toolset=mock_toolset)
        from litemind.media.types.media_action import Action

        action_blocks = [b for b in result.blocks if b.has_type(Action)]
        assert len(action_blocks) == 1


# =========================================================================
# _get_ollama_models_list tests
# =========================================================================


class TestGetOllamaModelsList:
    """Tests for _get_ollama_models_list thinking-variant generation."""

    def _make_client(self, model_names):
        """Create a mock Ollama client with the given model names."""
        client = MagicMock()
        models = []
        for name in model_names:
            m = MagicMock()
            m.model = name
            m.size = 1000
            models.append(m)
        client.list.return_value.models = models
        return client

    def test_thinking_variants_added(self):
        """Each regular model should get a -thinking variant."""
        from litemind.apis.providers.ollama.utils.list_models import (
            _get_ollama_models_list,
        )

        client = self._make_client(["llama3:8b"])
        result = _get_ollama_models_list(client)
        assert "llama3:8b" in result
        assert "llama3:8b-thinking" in result

    def test_no_double_thinking_suffix(self):
        """Models already ending in -thinking should not get a second suffix."""
        from litemind.apis.providers.ollama.utils.list_models import (
            _get_ollama_models_list,
        )

        client = self._make_client(["qwen3:30b-thinking"])
        result = _get_ollama_models_list(client)
        assert "qwen3:30b-thinking" in result
        assert "qwen3:30b-thinking-thinking" not in result
