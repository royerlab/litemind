from typing import TYPE_CHECKING

from arbol import aprint

if TYPE_CHECKING:
    from google.genai import Client

# Cache the result of the check_gemini_api_availability function
_cached_check_gemini_api_availability = None


def clear_availability_cache():
    """Clear the cached availability check result."""
    global _cached_check_gemini_api_availability
    _cached_check_gemini_api_availability = None


def check_gemini_api_availability(client: "Client"):
    """
    Check if the Gemini API is available by sending a test message to the API.

    Parameters
    ----------
    client : Client
        The google-genai Client instance to use for the test.

    Returns
    -------
    bool
        True if the API is available, False otherwise.
    """

    # Use the global variable
    global _cached_check_gemini_api_availability

    # Check if we have a cached result
    if _cached_check_gemini_api_availability is not None:
        return _cached_check_gemini_api_availability

    # We'll try a trivial call; if it fails, we assume invalid.
    try:
        # Minimal call: generate a short response
        resp = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents="Hello, What is your name? (short answer please)",
        )

        # If the response is not empty, we assume the API is available:
        result = len(resp.text) > 0

    except Exception as e:
        # If we get an error, we assume it's because the API is not available:
        aprint(f"Error while trying to check availability of Gemini API: {e}")

        import traceback

        traceback.print_exc()

        result = False

    if result:
        aprint("Gemini API is available.")
    else:
        aprint("Gemini API is not available.")

    # Cache the result:
    _cached_check_gemini_api_availability = result

    # Print the result:
    return result
