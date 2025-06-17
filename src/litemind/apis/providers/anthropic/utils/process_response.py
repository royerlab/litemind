from typing import Any, List, Optional, Union

from anthropic.types import WebSearchResultBlock
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.utils.json_to_object import json_to_object


def process_response_from_anthropic(
    anthropic_response: Any,
    response_format: Optional[Union[BaseModel, str]] = None,
) -> Message:
    """
    Process Anthropic response, handling all content block types including new built-in tools.

    Parameters
    ----------
    anthropic_response : Any
        The response from Anthropic API.
    response_format : Optional[BaseModel | str]
        The format of the response.

    Returns
    -------
    Message
        The response message to return.

    """

    # Initialize the processed response:
    processed_response = Message(role="assistant")

    # Check the stop reason:
    if anthropic_response.stop_reason == "refusal":
        # If the stop reason is refusal, return a message indicating refusal:
        processed_response.append_text("I cannot assist with that request (refusal).")

    # Text content:
    text_content = ""

    # Variable to hold whether the response is a tool use:
    is_tool_use = False

    # Create a litemind response message from Anthropic's response that includes tool calls:
    for block in anthropic_response.content:
        if block.type == "text":
            # Handle regular text blocks (may include citations)
            text = block.text
            processed_response.append_text(text)
            text_content += text

            # Handle citations if present in the block
            if hasattr(block, "citations") and block.citations:
                # Add citation information as metadata to the last text block
                last_block = processed_response.blocks[-1]
                if not hasattr(last_block, "attributes"):
                    last_block.attributes = {}

                citation_info = []
                for citation in block.citations:
                    if (
                        hasattr(citation, "type")
                        and citation.type == "web_search_result_location"
                    ):
                        citation_info.append(
                            {
                                "type": "web_search_citation",
                                "url": getattr(citation, "url", ""),
                                "title": getattr(citation, "title", ""),
                                "cited_text": getattr(citation, "cited_text", ""),
                            }
                        )

                if citation_info:
                    last_block.attributes["citations"] = citation_info

        elif block.type == "tool_use":
            # Handle regular function tool calls
            processed_response.append_tool_call(
                tool_name=block.name, arguments=block.input, id=block.id
            )
            is_tool_use = True

        elif block.type == "server_tool_use":
            # Handle built-in server tools (web search, MCP, code execution)
            # These are automatically executed by Anthropic, but we track them
            processed_response.append_tool_call(
                tool_name=block.name, arguments=block.input, id=block.id
            )
            is_tool_use = True

        elif block.type == "mcp_tool_use":
            # Handle MCP tool use (similar to server_tool_use)
            processed_response.append_tool_call(
                tool_name=block.name, arguments=block.input, id=block.id
            )
            is_tool_use = True

        elif block.type == "web_search_tool_result":
            # Handle web search results
            search_results: List[WebSearchResultBlock] = block.content

            processed_response.append_tool_use(
                tool_name="web_search",
                arguments={},
                result=search_results,
                id=block.tool_use_id,
            )

        elif block.type == "mcp_tool_result":
            # Handle MCP tool results
            from anthropic.types.beta import BetaTextBlock

            mcp_tool_results: List[BetaTextBlock] = block.content

            # Combine all text blocks into a single string:
            combined_text = "\n".join(
                [result.text for result in mcp_tool_results if result.type == "text"]
            )

            # Append the combined text to the processed response
            processed_response.append_tool_use(
                tool_name="mcp_tool",
                arguments={},
                result=combined_text,
                id=block.tool_use_id,
            )

        elif block.type == "thinking":
            # Handle thinking blocks (existing functionality)
            processed_response.append_thinking(
                block.thinking, signature=block.signature
            )

        elif block.type == "redacted_thinking":
            # Handle redacted thinking blocks (existing functionality)
            processed_response.append_thinking(block.data, redacted=True)

        else:
            raise ValueError(f"Unexpected block type: {block.type}")

    # Handle structured output conversion if requested
    if response_format and not is_tool_use:
        processed_response = json_to_object(
            processed_response, response_format, text_content
        )

    return processed_response
