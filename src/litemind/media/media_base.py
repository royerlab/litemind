from abc import ABC, abstractmethod
from typing import Any


class MediaBase(ABC):
    """Abstract base class for all media types in litemind.

    Media types include text, code, JSON, images, audio, video, documents,
    tables, and more. This class defines the interface that every media
    implementation must satisfy to integrate with the litemind API and
    message system.
    """

    @abstractmethod
    def get_content(self) -> Any:
        """Get the underlying content of this media.

        Returns
        -------
        Any
            The raw content. The concrete type depends on the media subclass
            (e.g., ``str`` for Text, ``dict`` for Json, a URI string for
            URI-based media).
        """
        pass

    @abstractmethod
    def to_message_block(self) -> "MessageBlock":
        """Convert this media into a MessageBlock for use in conversations.

        Returns
        -------
        MessageBlock
            A message block wrapping this media instance.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return a human-readable string representation of the media."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the media in characters for text or bytes for binary data."""
        pass
