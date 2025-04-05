import base64
import pickle
from typing import Type, TypeVar

T = TypeVar("T", bound="PickleSerializable")


class PickleSerializable:
    """Mixin class that provides automatic pickling serialization with base64 encoding support."""

    def to_pickle(self) -> bytes:
        """Convert object to pickled bytes."""
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls: Type[T], pickled_data: bytes) -> T:
        """Create instance from pickled bytes."""
        try:
            return pickle.loads(pickled_data)
        except (pickle.PickleError, TypeError, ValueError):
            raise ValueError("Invalid pickle data")

    def to_base64(self) -> str:
        """Convert object to base64-encoded string after pickling."""
        pickled_data = self.to_pickle()
        return base64.b64encode(pickled_data).decode("ascii")

    @classmethod
    def from_base64(cls: Type[T], base64_str: str) -> T:
        """Create instance from base64-encoded pickle data."""
        try:
            pickled_data = base64.b64decode(base64_str)
            return cls.from_pickle(pickled_data)
        except (base64.binascii.Error, TypeError, ValueError):
            raise ValueError("Invalid base64 pickle data")
