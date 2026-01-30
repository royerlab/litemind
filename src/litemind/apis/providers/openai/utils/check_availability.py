import os
from typing import Optional

from arbol import aprint

# Cache the result of the check_openai_api_availability function
_cached_check_openai_api_availability = None


def clear_availability_cache():
    """Clear the cached availability check result."""
    global _cached_check_openai_api_availability
    _cached_check_openai_api_availability = None


def check_openai_api_availability(client: Optional = None) -> bool:
    """
    Check if the OpenAI API key is valid by sending a test message using the Responses API.

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
        # If no client is provided, create a new OpenAI API client:
        if client is None:
            from openai import OpenAI

            # Initialize the OpenAI API client
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Call the OpenAI Responses API to verify the API key works:
        response = client.responses.create(
            model="gpt-4o-mini",  # Use a modern, cost-effective model
            input="Hello!",
            max_output_tokens=16,  # Minimum allowed value
        )
        # Check if the response contains output:
        result = len(response.output) > 0

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
