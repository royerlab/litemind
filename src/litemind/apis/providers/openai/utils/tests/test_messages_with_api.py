import os

from openai import OpenAI

from litemind.agent.messages.message import Message
from litemind.apis.providers.openai.utils.convert_messages import (
    convert_messages_for_openai,
)

__url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
__url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


def test_openai_api_text_only_message():
    # Create a text-only message
    message = Message(role="user")
    message.append_text("Hello, OpenAI!")

    # Convert to OpenAI format
    openai_messages = convert_messages_for_openai([message])

    # Initialize the OpenAI API client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=openai_messages, max_tokens=10
    )

    # Assert the API response contains expected structure
    assert "choices" in str(response)
    assert len(response.choices) > 0
    assert "assistant" in response.choices[0].message.role


def test_openai_api_text_and_image_message():
    # Create a message with text and image
    message = Message(role="user")
    message.append_text("What do you think of this image?")
    message.append_image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    )

    # Convert to OpenAI format
    openai_messages = convert_messages_for_openai([message])

    # Initialize the OpenAI API client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=openai_messages, max_tokens=10
    )

    # Assert the API response contains expected structure
    assert "choices" in str(response)
    assert len(response.choices) > 0
    assert "assistant" in response.choices[0].message.role


def test_openai_api_multiple_images():
    # Create a message with multiple images
    message = Message(role="user")
    message.append_text("What do you think of these images?")
    message.append_image(__url1)
    message.append_image(__url2)

    # Convert to OpenAI format
    openai_messages = convert_messages_for_openai([message])

    # Initialize the OpenAI API client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=openai_messages, max_tokens=10
    )

    # Assert the API response contains expected structure
    assert "choices" in str(response)
    assert len(response.choices) > 0
    assert "assistant" in response.choices[0].message.role


def test_openai_api_multiple_messages():
    # Create multiple messages
    message1 = Message(role="user")
    message1.append_text("Hello, OpenAI!")

    message2 = Message(role="user")
    message2.append_text("Here is an image, can you describe it?")
    message2.append_image(__url2)

    # Convert to OpenAI format
    openai_messages = convert_messages_for_openai([message1, message2])

    # Initialize the OpenAI API client
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Call the OpenAI API
    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=openai_messages, max_tokens=10
    )

    # Assert the API response contains expected structure
    assert "choices" in str(response)
    assert len(response.choices) > 0
    assert "assistant" in response.choices[0].message.role
