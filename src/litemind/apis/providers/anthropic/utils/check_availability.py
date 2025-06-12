from arbol import aprint

# Cache the result of the check_anthropic_api_availability function
_cached_check_anthropic_api_availability = None


def check_anthropic_api_availability(client: "Anthropic", model_name: str):
    # Use the global variable
    global _cached_check_anthropic_api_availability

    # Check if we have a cached result
    if _cached_check_anthropic_api_availability is not None:
        return _cached_check_anthropic_api_availability

    # We'll attempt a trivial request
    try:
        # Local import to avoid loading the library if not needed:
        from anthropic import Anthropic

        # Ensure the client is an instance of the Anthropic class:
        client: Anthropic = client

        # Create a test message:
        test_message = {
            "role": "user",
            "content": "Hello, wha is your name? (short answer please)",
        }

        # Call the Anthropic API to generate a completion:
        resp = client.messages.create(
            model=model_name,
            max_tokens=32,
            messages=[test_message],
        )

        # If no exception: assume it's valid enough
        _ = resp.content  # Accessing content to ensure it exists

        # If we get here, the API is available:
        result = True

    except Exception as e:
        # If we get an error, we assume it's because the API is not available:
        aprint(f"Error while trying to check availability of Anthropic API: {e}")

        # print stack trace:
        import traceback

        traceback.print_exc()

        result = False

    if result:
        aprint("Anthropic API is available.")
    else:
        aprint("Anthropic API is not available.")

    # Cache the result:
    _cached_check_anthropic_api_availability = result

    return result
