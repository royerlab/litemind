from arbol import aprint

# Cache the result of the check_ollama_api_availability function
_cached_check_ollama_api_availability = None


def clear_availability_cache():
    """Clear the cached availability check result."""
    global _cached_check_ollama_api_availability
    _cached_check_ollama_api_availability = None


def check_ollama_api_availability(client):
    """
    Check if the Ollama API is available by sending a test message to the API.

    Parameters
    ----------
    client : Any
        The Ollama API client instance.

    Returns
    -------
    bool
        True if the API is available, False otherwise.

    """

    global _cached_check_ollama_api_availability

    # Check if we have a cached result
    if _cached_check_ollama_api_availability is not None:
        return _cached_check_ollama_api_availability

    try:
        # Call the Ollama API to list the available models:
        client.list()

        # If we get here, the API is available:
        result = True
    except Exception:
        # Ollama server is not running or not reachable:
        result = False

    if result:
        aprint("Ollama API is available.")
    else:
        aprint("Ollama API is not available.")

    # Cache the result:
    _cached_check_ollama_api_availability = result

    return result
