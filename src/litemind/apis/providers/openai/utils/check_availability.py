from arbol import aprint

from litemind.agent.messages.message import Message
from litemind.apis.providers.openai.utils.convert_messages import (
    convert_messages_for_openai,
)

# Cache the result of the check_openai_api_availability function
_cached_check_openai_api_availability = None


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
    global _cached_check_openai_api_availability

    # Check if we have a cached result
    if _cached_check_openai_api_availability is not None:
        return _cached_check_openai_api_availability

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

    except Exception as e:
        # print error message from exception:
        aprint(f"Error while trying to check the OpenAI API key: {e}")

        # print stack trace:
        import traceback

        traceback.print_exc()

        result = False

    if result:
        aprint("OpenAI API is available.")
    else:
        aprint("OpenAI API is not available.")

    # Cache the result:
    _cached_check_openai_api_availability = result

    return result
