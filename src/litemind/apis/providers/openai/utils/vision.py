def has_vision_support(model_name: str) -> bool:
    """
    Tests if an OpenAI model supports vision by attempting to send a minimal valid image.

    Args:
        model_name (str): Name of the model to tests.

    Returns:
        bool: True if the model supports vision, False otherwise

    Raises:
        ValueError: If model_name is empty or invalid
    """
    import openai

    # Cleanup model name:
    model_name = model_name.strip()

    if not isinstance(model_name, str) or len(model_name) == 0:
        raise ValueError("Model name must be a non-empty string")

    # Create a 1x1 transparent pixel in base64
    minimal_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

    try:
        # Attempt to create a vision message
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{minimal_image}"},
                },
            ],
        }

        # Try to create a chat completion with the image

        response = openai.chat.completions.create(
            model=model_name, messages=[message], max_tokens=1  # Minimize token usage
        )
        return True

    except openai.BadRequestError as e:
        # If we get an error about images not being supported, return False
        if (
            (
                "does not support vision" in str(e).lower()
                or "invalid message format" in str(e).lower()
            )
            or "image_url is only supported by certain models." in str(e).lower()
            or "you must provide a model parameter" in str(e).lower()
        ):
            return False
        # For other bad request errors, re-raise
        raise

    except openai.NotFoundError as e:
        # For other OpenAI errors (API key, network, etc.), re-raise
        return False

    except openai.OpenAIError as e:
        # For other OpenAI errors (API key, network, etc.), re-raise
        raise
