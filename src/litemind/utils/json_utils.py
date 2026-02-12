"""JSON serialization and deserialization utilities.

Provides a mixin class for automatic JSON serialization of Python objects,
with support for nested objects, dates, and sets.
"""

import datetime
import inspect
import json
from typing import Any, Dict, Optional


class JSONSerializable:
    """
    Mixin class providing automatic JSON serialization.

    Supports recursive handling of nested ``JSONSerializable`` objects,
    lists, and dictionaries. Skips private attributes (prefixed with ``_``).
    """

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serialize this object to a JSON string.

        Parameters
        ----------
        indent : int, optional
            Number of spaces for JSON indentation. None for compact output.

        Returns
        -------
        str
            The JSON string representation.
        """
        return json.dumps(self.to_dict(), cls=_JSONEncoder, indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert this object to a dictionary.

        Recursively converts nested ``JSONSerializable`` objects. Skips
        private attributes (those starting with ``_``).

        Returns
        -------
        dict
            Dictionary representation of this object.
        """
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue

            # Handle nested JSONSerializable objects
            if isinstance(v, JSONSerializable):
                result[k] = v.to_dict()
            # Handle lists of objects
            elif isinstance(v, list):
                result[k] = [
                    item.to_dict() if isinstance(item, JSONSerializable) else item
                    for item in v
                ]
            # Handle dictionaries
            elif isinstance(v, dict):
                result[k] = {
                    key: (
                        value.to_dict()
                        if isinstance(value, JSONSerializable)
                        else value
                    )
                    for key, value in v.items()
                }
            else:
                result[k] = v
        return result

    @classmethod
    def from_json(cls, json_str: str) -> "JSONSerializable":
        """
        Create an instance from a JSON string.

        Parameters
        ----------
        json_str : str
            A valid JSON string.

        Returns
        -------
        JSONSerializable
            A new instance populated from the JSON data.

        Raises
        ------
        ValueError
            If the string is not valid JSON.
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSONSerializable":
        """
        Create an instance from a dictionary.

        Matches dictionary keys to constructor parameters and sets any
        remaining keys as attributes on the instance.

        Parameters
        ----------
        data : dict
            Dictionary with keys corresponding to the class attributes.

        Returns
        -------
        JSONSerializable
            A new instance populated from the dictionary.
        """
        # Extract constructor parameters
        init_signature = inspect.signature(cls.__init__)
        param_names = set(list(init_signature.parameters.keys())[1:])  # Exclude 'self'

        # Build constructor arguments
        kwargs = {}
        for param in param_names:
            if param in data:
                value = data[param]

                # Special handling for nested objects
                if param == "child" and isinstance(value, dict) and "name" in value:
                    kwargs[param] = cls.from_dict(value)
                else:
                    kwargs[param] = value

        # Create instance
        instance = cls(**kwargs)

        # Set remaining attributes (that weren't passed to constructor)
        for k, v in data.items():
            if k not in kwargs:
                setattr(instance, k, v)

        return instance


class _JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles dates, sets, and objects with ``to_dict``.

    Extends the standard ``json.JSONEncoder`` to serialize
    ``datetime.datetime``, ``datetime.date``, ``set``, and any object
    that provides a ``to_dict()`` method.
    """

    def default(self, obj: Any) -> Any:
        """Return a serializable representation of the given object.

        Parameters
        ----------
        obj : Any
            The object to serialize.

        Returns
        -------
        Any
            A JSON-serializable representation: ISO-format string for
            dates, list for sets, dict for objects with ``to_dict()``,
            or delegates to the parent encoder.
        """
        # Handle dates and times
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()

        # Handle sets
        if isinstance(obj, set):
            return list(obj)

        # Handle objects with to_dict method
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()

        return super().default(obj)
