from functools import lru_cache

from arbol import aprint

from litemind.agent.messages.message import Message
from litemind.apis.providers.openai.utils.convert_messages import (
    convert_messages_for_openai,
)


@lru_cache
def check_openai_api_availability(client):
    """
    Check if the OpenAI API key is valid by sending a test message to the API.

    Parameters
    ----------
    client: OpenAI API client

    Returns
    -------
    bool
        True if the API key is valid, False otherwise.

    """

    try:
        messages = []
        # Create a user message:
        user_message = Message(role="user")
        user_message.append_text("Hello!")
        messages.append(user_message)
        # Call the OpenAI raw API to generate a completion (not using the API wrapper):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=convert_messages_for_openai(messages),
            max_completion_tokens=10,
        )
        # Check if the response contains any choices:
        result = len(response.choices) > 0
        if result:
            aprint("OpenAI API key is valid.")
        else:
            aprint("OpenAI API key is invalid.")

    except Exception as e:
        # print error message from exception:
        aprint(f"Error while trying to check the OpenAI API key: {e}")

        # print stack trace:
        import traceback

        traceback.print_exc()

        result = False

    return result
