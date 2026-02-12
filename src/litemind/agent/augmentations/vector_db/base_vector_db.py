"""Abstract base class for vector database implementations.

This module defines ``BaseVectorDatabase``, the interface that all vector
database backends must implement. Vector databases store ``Information``
objects alongside their vector embeddings and enable semantic similarity
search, forming the storage backbone for Retrieval-Augmented Generation
(RAG) augmentations in litemind.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

from pydantic import BaseModel

from litemind.agent.augmentations.information.information import Information
from litemind.media.media_base import MediaBase


class BaseVectorDatabase(ABC):
    """
    Abstract base class for vector databases.

    Vector databases store informations along with their vector embeddings to enable
    semantic similarity search.
    """

    @abstractmethod
    def add_informations(
        self,
        informations: List[Information],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add informations to the vector database.

        Parameters
        ----------
        informations : List[Information]
            The informations to add to the database.
        embeddings : Optional[List[List[float]]]
            Pre-computed embedding vectors. If not provided, embeddings
            are computed automatically.

        Returns
        -------
        List[str]
            The IDs of the added informations.
        """
        pass

    @abstractmethod
    def get_information(self, information_id: str) -> Optional[Information]:
        """
        Retrieve an information by its ID.

        Parameters
        ----------
        information_id : str
            The ID of the information to retrieve.

        Returns
        -------
        Optional[Information]
            The information, or None if not found.
        """
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: Union[str, BaseModel, MediaBase, Information, List[float]],
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Information]:
        """
        Find informations most similar to the query.

        Parameters
        ----------
        query : Union[str, BaseModel, MediaBase, Information, List[float]]
            The query string, media object, Information, or a raw
            embedding vector.
        k : int
            The maximum number of results to return.
        threshold : float
            Minimum similarity score. Results below this value are
            excluded.

        Returns
        -------
        List[Information]
            The most similar informations, ordered by descending
            similarity score.
        """
        pass

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete informations from the database.

        Parameters
        ----------
        ids: List[str]
            The IDs of the informations to delete.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """
        Clear all informations from the database.
        """
        pass

    @abstractmethod
    def save(self) -> None:
        """
        Persist the database state to storage.

        Some implementations (e.g., Qdrant) persist automatically,
        making this a no-op. Others (e.g., in-memory) require an
        explicit call to save. It is recommended to call this after
        any modification to ensure portability across implementations.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the database connection.

        This is optional and may not be needed for all implementations.
        """
        pass
