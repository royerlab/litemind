from types import SimpleNamespace
from typing import Any, Dict, List, Union


def recursive_dict_to_attributes(
    obj: Union[Dict[str, Any], List[Any], Any]
) -> Union[SimpleNamespace, List[Any], Any]:
    """
    Recursively convert a dictionary to an object with attributes.

    Parameters
    ----------
    obj : Union[Dict[str, Any], List[Any], Any]
        The object to convert. Can be a dict, list, or any other type.

    Returns
    -------
    Union[SimpleNamespace, List[Any], Any]
        If input is a dict, returns a SimpleNamespace with attributes.
        If input is a list, returns a list with converted elements.
        Otherwise, returns the input unchanged.
    """
    if isinstance(obj, dict):
        # Recursively convert all values in the dict
        return SimpleNamespace(
            **{k: recursive_dict_to_attributes(v) for k, v in obj.items()}
        )
    elif isinstance(obj, list):
        # Recursively convert each element in the list
        return [recursive_dict_to_attributes(item) for item in obj]
    else:
        return obj
