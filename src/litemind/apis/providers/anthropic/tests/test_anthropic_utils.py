"""Unit tests for Anthropic message conversion and response processing.

Tests cover the empty text block handling added to:
- convert_messages_for_anthropic (skip empty text blocks)
- process_response_from_anthropic (skip empty text blocks from API responses)
"""

from unittest.mock import MagicMock

from litemind.agent.messages.message import Message
from litemind.apis.providers.anthropic.utils.convert_messages import (
    convert_messages_for_anthropic,
)
from litemind.apis.providers.anthropic.utils.process_response import (
    process_response_from_anthropic,
)
from litemind.media.types.media_text import Text

# =========================================================================
# convert_messages_for_anthropic — empty text block tests
# =========================================================================


class TestConvertMessagesEmptyTextBlocks:
    """Empty text blocks must be skipped to avoid Anthropic 400 errors."""

    def test_empty_assistant_text_skipped(self):
        """Assistant message with only whitespace text should not produce empty blocks."""
        msg = Message(role="assistant")
        msg.append_text("   ")  # whitespace only — becomes empty after rstrip
        result = convert_messages_for_anthropic([msg])
        # The message should be dropped or have no text content blocks:
        for m in result:
            if m["role"] == "assistant":
                for block in m.get("content", []):
                    if isinstance(block, dict) and block.get("type") == "text":
                        assert block[
                            "text"
                        ], "Empty text block should have been skipped"

    def test_nonempty_text_preserved(self):
        """Non-empty text blocks should still be included."""
        msg = Message(role="user", text="Hello")
        result = convert_messages_for_anthropic([msg])
        assert len(result) == 1
        content = result[0]["content"]
        text_blocks = [
            b for b in content if isinstance(b, dict) and b.get("type") == "text"
        ]
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "Hello"

    def test_mixed_empty_and_nonempty(self):
        """Only non-empty text blocks should survive conversion."""
        msg = Message(role="assistant")
        msg.append_text("")  # empty
        msg.append_text("Valid text")
        result = convert_messages_for_anthropic([msg])
        for m in result:
            if m["role"] == "assistant":
                text_blocks = [
                    b
                    for b in m.get("content", [])
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                assert all(b["text"] for b in text_blocks)
                assert any("Valid text" in b["text"] for b in text_blocks)


# =========================================================================
# process_response_from_anthropic — empty text block tests
# =========================================================================


class TestProcessResponseEmptyTextBlocks:
    """Empty text blocks from API responses must be skipped."""

    def _make_response(self, content_blocks):
        """Create a mock Anthropic response with given content blocks."""
        resp = MagicMock()
        resp.content = content_blocks
        resp.stop_reason = "end_turn"
        resp.usage = MagicMock()
        resp.usage.input_tokens = 10
        resp.usage.output_tokens = 5
        resp.usage.cache_creation_input_tokens = 0
        resp.usage.cache_read_input_tokens = 0
        return resp

    def _text_block(self, text):
        block = MagicMock()
        block.type = "text"
        block.text = text
        return block

    def test_empty_text_block_skipped(self):
        """Empty text blocks from the API should not produce text in the message."""
        resp = self._make_response([self._text_block("")])
        result = process_response_from_anthropic(resp)
        text_blocks = [
            b for b in result.blocks if b.has_type(Text) and not b.is_thinking()
        ]
        assert len(text_blocks) == 0

    def test_nonempty_text_block_preserved(self):
        """Non-empty text blocks should still be processed."""
        resp = self._make_response([self._text_block("Hello world")])
        result = process_response_from_anthropic(resp)
        text_blocks = [
            b for b in result.blocks if b.has_type(Text) and not b.is_thinking()
        ]
        assert len(text_blocks) == 1
        assert "Hello world" in text_blocks[0].get_content()

    def test_mixed_empty_and_nonempty_response(self):
        """Only non-empty text blocks should appear in the processed message."""
        resp = self._make_response(
            [
                self._text_block(""),
                self._text_block("Valid response"),
            ]
        )
        result = process_response_from_anthropic(resp)
        text_blocks = [
            b for b in result.blocks if b.has_type(Text) and not b.is_thinking()
        ]
        assert len(text_blocks) == 1
        assert "Valid response" in text_blocks[0].get_content()
