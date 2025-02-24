from typing import Type, TypeVar


def extract_json_substring(json_str: str) -> str:
    """
    Extract a JSON substring from a string.

    Parameters
    ----------

    json_str : str
        The string to extract the JSON substring from.

    Returns
    -------
    str
        The JSON substring.

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
    Use Pydantic to parse a JSON string into a given class.

    Parameters
    ----------
    json_str: str
        The JSON string to parse.
    clazz: Type[T]
        The class to parse the JSON string into.

    Returns
    -------

    """

    # Cleanup the JSON string:
    json_str = extract_json_substring(json_str)

    # Parse the JSON string:
    return clazz.model_validate_json(json_str)
