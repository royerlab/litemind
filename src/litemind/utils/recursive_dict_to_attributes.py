from types import SimpleNamespace


def recursive_dict_to_attributes(obj):
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
