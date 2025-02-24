from litemind.agent.messages.message import Message
from litemind.apis.providers.openai.utils.convert_messages import (
    convert_messages_for_openai,
)


def test_convert_text_only_message():
    # Create a message with only text
    message = Message(role="user")
    message.append_text("Hello, world!")

    # Convert to OpenAI format
    converted = convert_messages_for_openai([message])

    # Assert the structure is as expected
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == [{"type": "text", "text": "Hello, world!"}]


def test_convert_image_only_message():
    # Create a message with only an image URL
    message = Message(role="user")
    message.append_image("https://example.com/image.png")

    # Convert to OpenAI format
    converted = convert_messages_for_openai([message])

    # Assert the structure is as expected
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == [
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    ]


def test_convert_text_and_image_message():
    # Create a message with both text and an image URL
    message = Message(role="user")
    message.append_text("Here is an image:")
    message.append_image("https://example.com/image.png")

    # Convert to OpenAI format
    converted = convert_messages_for_openai([message])

    # Assert the structure is as expected
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == [
        {"type": "text", "text": "Here is an image:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]


def test_convert_multiple_images_message():
    # Create a message with multiple image URLs
    message = Message(role="user")
    message.append_text("Here are two images:")
    message.append_image("https://example.com/image1.png")
    message.append_image("https://example.com/image2.png")

    # Convert to OpenAI format
    converted = convert_messages_for_openai([message])

    # Assert the structure is as expected
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == [
        {"type": "text", "text": "Here are two images:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image1.png"}},
        {"type": "image_url", "image_url": {"url": "https://example.com/image2.png"}},
    ]


def test_convert_empty_message():
    # Create an empty message (no text, no images)
    message = Message(role="user")

    # Convert to OpenAI format
    converted = convert_messages_for_openai([message])

    # Assert the structure is as expected
    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == []


def test_convert_multiple_messages():
    # Create multiple messages with various contents
    message1 = Message(role="user")
    message1.append_text("Hello, world!")

    message2 = Message(role="assistant")
    message2.append_text("Here is an image:")
    message2.append_image("https://example.com/image.png")

    # Convert to OpenAI format
    converted = convert_messages_for_openai([message1, message2])

    # Assert the structure is as expected
    assert len(converted) == 2
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == [{"type": "text", "text": "Hello, world!"}]
    assert converted[1]["role"] == "assistant"
    assert converted[1]["content"] == [
        {"type": "text", "text": "Here is an image:"},
        {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
    ]
