import base64
import pickle
from typing import Type, TypeVar

T = TypeVar("T", bound="PickleSerializable")


class PickleSerializable:
    """
    Mixin class providing pickle-based serialization with base64 support.

    Allows objects to be serialized to/from pickle bytes or base64-encoded
    strings for transport or storage.
    """

    def to_pickle(self) -> bytes:
        """
        Serialize this object to pickle bytes.

        Returns
        -------
        bytes
            The pickled representation of this object.
        """
        return pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_pickle(cls: Type[T], pickled_data: bytes) -> T:
        """
        Deserialize an instance from pickle bytes.

        Parameters
        ----------
        pickled_data : bytes
            The pickled data to deserialize.

        Returns
        -------
        T
            The deserialized object.

        Raises
        ------
        ValueError
            If the data cannot be unpickled.
        """
        try:
            return pickle.loads(pickled_data)
        except (pickle.PickleError, TypeError, ValueError):
            raise ValueError("Invalid pickle data")

    def to_base64(self) -> str:
        """
        Serialize this object to a base64-encoded string.

        Returns
        -------
        str
            A base64 ASCII string of the pickled object.
        """
        pickled_data = self.to_pickle()
        return base64.b64encode(pickled_data).decode("ascii")

    @classmethod
    def from_base64(cls: Type[T], base64_str: str) -> T:
        """
        Deserialize an instance from a base64-encoded pickle string.

        Parameters
        ----------
        base64_str : str
            The base64-encoded pickle data.

        Returns
        -------
        T
            The deserialized object.

        Raises
        ------
        ValueError
            If the data cannot be decoded or unpickled.
        """
        try:
            pickled_data = base64.b64decode(base64_str)
            return cls.from_pickle(pickled_data)
        except (base64.binascii.Error, TypeError, ValueError):
            raise ValueError("Invalid base64 pickle data")
