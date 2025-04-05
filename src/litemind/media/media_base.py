from abc import ABC, abstractmethod
from typing import Any


class MediaBase(ABC):
    """
    Abstract base class defining the interface for 'media' which can be text, code, json, objects, images, audios, videos, etc...

    This class defines the minimum contract that any media implementation must fulfill
    to work with the rest of the litemind API
    """

    @abstractmethod
    def get_content(self) -> Any:
        """
        Get the content of the media.

        Returns
        -------
        Any
            The content of the media.
        """
        pass

    @abstractmethod
    def to_message_block(self) -> "MessageBlock":
        """
        Convert this media to a message block.

        Returns
        -------
        MessageBlock
            The media formatted as a message block.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the media."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Length of the media in bytes for binary data and characters for text"""
        pass
