def is_openai_reasoning_model(model_name: str) -> bool:
    """
    Return True iff `model_name` is one of OpenAI’s public reasoning models.

    >>> is_openai_reasoning_model("o3")
    True
    >>> is_openai_reasoning_model("gpt-4.5-preview")
    False
    """
    return (
        "o1" in model_name
        or "o3" in model_name
        or "o4" in model_name
        or "o5" in model_name
    )
