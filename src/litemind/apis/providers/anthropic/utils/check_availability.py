from functools import lru_cache


@lru_cache
def check_anthropic_api_availability(client: "Anthropic", model_name: str):
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

    except Exception:
        # If we get an error, we assume it's because the API is not available:
        import traceback

        traceback.print_exc()
        result = False

    return result
