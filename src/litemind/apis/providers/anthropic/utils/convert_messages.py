from typing import List, Optional

from pydantic import BaseModel

from litemind.agent.messages.actions.action_base import ActionBase
from litemind.agent.messages.actions.tool_call import ToolCall
from litemind.agent.messages.actions.tool_use import ToolUse
from litemind.agent.messages.message import Message
from litemind.media.types.media_action import Action
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text


def convert_messages_for_anthropic(
    messages: List[Message],
    response_format: Optional[BaseModel] = None,
    cache_support: bool = True,
    media_converter: Optional["MediaConverter"] = None,
) -> List["MessageParam"]:
    """
    Convert litemind Messages into Anthropic's MessageParam format with support for new built-in tools:
        [
          {"role": "user", "content": [{"type": "text", "text": "Hello!"}, {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}]},
          {"role": "assistant", "content": [{"type": "text", "text": "Hello there!"}]},
          ...
        ]
    """

    from anthropic.types import MessageParam

    anthropic_messages: List[MessageParam] = []

    for message in messages:
        content = []

        for block in message.blocks:
            if block.has_type(Text) and not block.is_thinking():

                # Get Text's string:
                text: str = block.get_content()

                if message.role == "assistant":
                    # remove trailing whitespace as it is not allowed by Anthropic!
                    text = text.rstrip()

                # Create text block
                text_block = {"type": "text", "text": text}

                # Handle citations if they exist in block attributes
                if (
                    hasattr(block, "attributes")
                    and block.attributes
                    and "citations" in block.attributes
                ):
                    citations = block.attributes["citations"]
                    if citations:
                        # Convert citations to Anthropic format
                        anthropic_citations = []
                        for citation in citations:
                            if citation.get("type") == "web_search_citation":
                                anthropic_citations.append(
                                    {
                                        "type": "web_search_result_location",
                                        "url": citation.get("url", ""),
                                        "title": citation.get("title", ""),
                                        "cited_text": citation.get("cited_text", ""),
                                    }
                                )

                        if anthropic_citations:
                            text_block["citations"] = anthropic_citations

                content.append(text_block)

            elif block.has_type(Text) and block.is_thinking():

                # Get Text's string:
                text: str = block.get_content()

                if block.is_redacted():
                    # if the block is redacted, add it as a redacted thinking block:
                    content.append({"type": "redacted_thinking", "data": text})
                else:
                    # Append the thinking text to the Anthropic's message content:
                    content.append(
                        {
                            "type": "thinking",
                            "thinking": text,
                            "signature": block.attributes.get("signature", ""),
                        }
                    )

            elif block.has_type(Image):

                # Cast to Image:
                image: Image = block.media

                # Get media type
                media_type = image.get_media_type()

                # get base64 data:
                base64_data = image.to_base64_data()

                # Add the image to the content:
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": (base64_data),
                        },
                    }
                )

            elif block.has_type(Document):

                # Cast to Document:
                document: Document = block.media

                if document.has_extension("pdf"):

                    # Get media type
                    media_type = document.get_media_type()

                    # get base64 data:
                    base64_data = document.to_base64_data()

                    # Add the document to the content:
                    content.append(
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_data,
                            },
                        }
                    )
                else:
                    # Convert Document to text:
                    markdown: str = document.to_markdown(
                        media_converter=media_converter
                    )

                    # Add the document text to the content:
                    content.append({"type": "text", "text": markdown})

            elif block.has_type(Action):

                # Get the tool action:
                tool_action: ActionBase = block.get_content()

                # if contents is a ToolUse object do the following:
                if isinstance(tool_action, ToolUse):

                    # Get the tool use object:
                    tool_use: ToolUse = tool_action

                    # Add the tool result to the content:
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": [
                                {
                                    "type": "text",
                                    "text": tool_use.result,
                                }
                            ],
                        }
                    )

                elif isinstance(tool_action, ToolCall):

                    # Get the tool call object:
                    tool_call: ToolCall = tool_action

                    # Add regular tool_use block
                    content.append(
                        {
                            "id": tool_call.id,
                            "type": "tool_use",
                            "name": tool_call.tool_name,
                            "input": tool_call.arguments,
                        }
                    )

                else:
                    raise ValueError(
                        f"Unsupported action type: {type(tool_action).__name__}"
                    )

            else:
                raise ValueError(
                    f"Unsupported block type: {type(block.media).__name__}"
                )

        anthropic_messages.append(
            {
                "role": message.role,
                "content": content,
            }
        )

    # Add cache_control to the last message if requested:
    if cache_support:
        if anthropic_messages and anthropic_messages[-1]["content"]:
            anthropic_messages[-1]["content"][-1]["cache_control"] = {
                "type": "ephemeral"
            }

    # Add prompt that specifies that response should be in JSON following given JSON schema:
    if response_format:
        anthropic_messages[-1]["content"].append(
            {
                "type": "text",
                "text": f"\nPlease provide answer as a JSON string that adheres to the following schema:\n{response_format.model_json_schema()}\n\n Without text before or after.",
            }
        )

    return anthropic_messages
