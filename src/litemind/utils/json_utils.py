import datetime
import inspect
import json
from typing import Any, Dict, Optional


class JSONSerializable:
    """Mixin class that provides automatic JSON serialization with recursive support."""

    def to_json(self, indent: Optional[int] = None) -> str:
        """Convert object to JSON string."""
        return json.dumps(self.to_dict(), cls=_JSONEncoder, indent=indent)

    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary with recursive handling of nested objects."""
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
        """Create instance from JSON string."""
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON string")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JSONSerializable":
        """Create instance from dictionary with recursive handling."""
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
    """Custom JSON encoder that handles common non-serializable types."""

    def default(self, obj: Any) -> Any:
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
