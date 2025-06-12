from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Type

from litemind.agent.messages.message_block import MessageBlock
from litemind.media.media_base import MediaBase


class InformationBase(ABC):
    """
    Abstract base class defining the interface for 'pieces of information' used in augmentations.

    This class defines the minimum contract that any document implementation must fulfill
    to work with the augmentation system.
    """

    @property
    @abstractmethod
    def content(self) -> MediaBase:
        """The  content of the information"""
        pass

    @property
    @abstractmethod
    def type(self) -> Type[MediaBase]:
        """
        The type of the information.

        Returns
        -------
        Type[MediaBase]
            The type of the information.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> dict:
        """
        Metadata associated with the information.

        Returns
        -------
        dict
            A dictionary containing metadata about the information.
        """
        pass

    @abstractmethod
    def has_type(self, media_type: Type[MediaBase]) -> bool:
        """
        Check if the information has a specific type.

        Parameters
        ----------
        media_type: str
            The type to check against.

        Returns
        -------
        bool
            True if the information has the specified type, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def id(self) -> str:
        """A unique identifier for the information."""
        pass

    @property
    @abstractmethod
    def score(self) -> Optional[float]:
        """A relevance score, usually assigned during retrieval."""
        pass

    @abstractmethod
    def compute_embedding(
        self, embedding_function: Optional[Callable[[str], List[float]]] = None
    ) -> List[float]:
        """
        Compute an embedding for this piece of information.
        If an embedding function is provided, it will be used to compute the embedding.
        Otherwise, the embedding will be computed using LiteMind's default embedding function.

        Parameters
        ----------
        embedding_function: Optional[Callable[[str], List[float]]]
            A function that takes a string and returns an embedding vector.

        Returns
        -------
        List[float]
            The computed embedding.
        """
        pass

    @abstractmethod
    def to_message_block(self) -> MessageBlock:
        """
        Convert the piece of information to a message block format.

        Returns
        -------
        MessageBlock
            The piece of information formatted as a message block.
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        """String representation of the piece of information."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Length of the piece of information"""
        pass
