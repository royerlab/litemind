from arbol import aprint

# Cache the result of the check_gemini_api_availability function
_cached_check_gemini_api_availability = None


def check_gemini_api_availability(api_key):
    """
    Check if the Gemini API is available by sending a test message to the API.
    Parameters
    ----------
    api_key: str
        API key to use for the test.

    Returns
    -------

    """

    # Use the global variable
    global _cached_check_gemini_api_availability

    # Check if we have a cached result
    if _cached_check_gemini_api_availability is not None:
        return _cached_check_gemini_api_availability

    # Local import to avoid loading the library if not needed:
    import google.generativeai as genai

    # We’ll try a trivial call; if it fails, we assume invalid.
    try:
        # Check if the API key is provided:
        if api_key is not None:
            genai.configure(api_key=api_key)
        else:
            pass

        # Minimal call: generate a short response
        model = genai.GenerativeModel()

        # Generate a short response:
        resp = model.generate_content("Hello, What is yur name? (short answer please)")

        # If the response is not empty, we assume the API is available:
        result = len(resp.text) > 0

    except Exception as e:
        # If we get an error, we assume it's because the API is not available:
        aprint(f"Error while trying to check availability of Ollama API: {e}")

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
