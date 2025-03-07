from functools import lru_cache


@lru_cache
def check_gemini_api_availability(api_key, default_api_key):
    """
    Check if the Gemini API is available by sending a test message to the API.
    Parameters
    ----------
    api_key: str
        API key to use for the test.
    default_api_key: str
        Default API key to reset to after the test.

    Returns
    -------

    """

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

    except Exception:
        # If we get an error, we assume it's because the API is not available:
        import traceback

        traceback.print_exc()
        result = False

    # Print the result:
    return result
