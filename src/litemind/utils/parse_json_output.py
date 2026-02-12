"""Utilities for extracting and parsing JSON from text output."""

from typing import Type, TypeVar


def extract_json_substring(json_str: str) -> str:
    """
    Extract the outermost JSON object from a string.

    Finds the first ``{`` and last ``}`` in the string and returns the
    substring between them (inclusive).

    Parameters
    ----------
    json_str : str
        The string containing a JSON object.

    Returns
    -------
    str
        The extracted JSON substring.
    """

    # Find the first opening brace:
    start = json_str.find("{")

    # Find the last closing brace:
    end = json_str.rfind("}")

    # Extract the JSON substring:
    json_substring = json_str[start : end + 1]

    return json_substring


T = TypeVar("T")


def parse_json(json_str: str, clazz: Type[T]) -> T:
    """
    Parse a JSON string into a Pydantic model instance.

    Extracts the outermost JSON object from the string (between the
    first ``{`` and last ``}``) and validates it against the given class.

    Parameters
    ----------
    json_str : str
        A string containing a JSON object.
    clazz : Type[T]
        A Pydantic model class to parse the JSON into.

    Returns
    -------
    T
        An instance of ``clazz`` populated from the JSON data.
    """

    # Cleanup the JSON string:
    json_str = extract_json_substring(json_str)

    # Parse the JSON string:
    return clazz.model_validate_json(json_str)
