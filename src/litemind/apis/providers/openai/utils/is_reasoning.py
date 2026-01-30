import re


def is_openai_reasoning_model(model_name: str) -> bool:
    """
    Return True iff `model_name` is one of OpenAI's public reasoning models.

    Matches o-series models: o1, o3, o4, o5, o1-mini, o3-high, o4-mini-low, etc.
    Does NOT match models like 'gpt-4o1' or 'modelo3'.

    >>> is_openai_reasoning_model("o3")
    True
    >>> is_openai_reasoning_model("o3-high")
    True
    >>> is_openai_reasoning_model("o4-mini-low")
    True
    >>> is_openai_reasoning_model("gpt-4.5-preview")
    False
    >>> is_openai_reasoning_model("gpt-4o")
    False
    """
    # Match o-series models: starts with o followed by 1,3,4,5 then either end or hyphen
    return bool(re.match(r"^o[1345](-|$)", model_name))


def does_openai_model_support_temperature(model_name: str) -> bool:
    """
    Return True iff `model_name` supports the temperature parameter.

    Models that do NOT support temperature:
    - O-series reasoning models (o1, o3, o4, o5, etc.)
    - GPT-5.x series models (gpt-5, gpt-5.1, gpt-5.2, etc.)

    >>> does_openai_model_support_temperature("gpt-4o")
    True
    >>> does_openai_model_support_temperature("gpt-4o-mini")
    True
    >>> does_openai_model_support_temperature("gpt-4.1")
    True
    >>> does_openai_model_support_temperature("o3")
    False
    >>> does_openai_model_support_temperature("o3-high")
    False
    >>> does_openai_model_support_temperature("gpt-5")
    False
    >>> does_openai_model_support_temperature("gpt-5.2-pro")
    False
    >>> does_openai_model_support_temperature("gpt-5.1-codex")
    False
    """
    # O-series reasoning models don't support temperature
    if is_openai_reasoning_model(model_name):
        return False

    # GPT-5.x series models don't support temperature
    # Matches: gpt-5, gpt-5.1, gpt-5.2, gpt-5-pro, gpt-5.2-pro, gpt-5.1-codex, etc.
    if re.match(r"^gpt-5(\.\d+)?(-|$)", model_name):
        return False

    return True
