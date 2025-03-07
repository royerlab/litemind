from functools import lru_cache

from arbol import aprint


@lru_cache
def check_ollama_api_availability(client):
    """
    Check if the Ollama API is available by sending a test message to the API.

    Parameters
    ----------
    client:
        Ollama API client


    Returns
    -------
    bool
        True if the API is available, False otherwise.

    """
    try:
        # Call the Ollama API to list the available models:
        client.list()

        # If we get here, the API is available:
        result = True
    except Exception:
        # If we get an error, we assume it's because Ollama is not running:
        result = False

    if result:
        aprint("Ollama server is available.")
    else:
        aprint("Ollama server is not available.")

    return result
