"""Utilities for identifying OpenAI reasoning models and their parameter constraints.

Provides helper functions to determine whether a given OpenAI model name
corresponds to an o-series reasoning model (o1, o3, o4, o5) and whether
the model supports the ``temperature`` parameter.
"""

import re


def is_openai_reasoning_model(model_name: str) -> bool:
    """Check if a model is an OpenAI reasoning (o-series) model.

    Matches o-series models: o1, o3, o4, o5 and their variants
    (e.g., o1-mini, o3-high, o4-mini-low). Does NOT match models
    like ``gpt-4o1`` or ``modelo3``.

    Parameters
    ----------
    model_name : str
        The model name to check.

    Returns
    -------
    bool
        True if the model is an o-series reasoning model.

    Examples
    --------
    >>> is_openai_reasoning_model("o3")
    True
    >>> is_openai_reasoning_model("gpt-4o")
    False
    """
    # Match o-series models: starts with o followed by 1,3,4,5 then either end or hyphen
    return bool(re.match(r"^o[1345](-|$)", model_name))


def does_openai_model_support_temperature(model_name: str) -> bool:
    """Check if a model supports the temperature parameter.

    O-series reasoning models (o1, o3, o4, o5) and GPT-5.x series
    models do not support the temperature parameter.

    Parameters
    ----------
    model_name : str
        The model name to check.

    Returns
    -------
    bool
        True if the model supports temperature, False otherwise.

    Examples
    --------
    >>> does_openai_model_support_temperature("gpt-4o")
    True
    >>> does_openai_model_support_temperature("o3")
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
