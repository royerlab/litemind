def is_vision_model(model_name: str) -> bool:
    # Then, we check if it is a vision model:
    if 'vision' in model_name or 'gpt-4o' in model_name or 'gpt-o1' in model_name:
        return True

    # then, we check if it is an audio model:
    if 'audio' in model_name:
        return False

    # Any other model is not a vision model:
    return False


def is_tool_model(model_name: str) -> bool:
    # Only old models don't support tools:
    if 'gpt-3.5' in model_name:
        return False

    # Any other model supports tools:
    return True
