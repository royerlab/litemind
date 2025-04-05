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
        informations: List[Document]
            The informations to add to the database.
        embeddings: Optional[List[List[float]]]
            The embeddings for the informations. If not provided, they will be computed.

        Returns
        -------
        List[str]
            The IDs of the added informations.
        """
        pass

    @abstractmethod
    def get_information(self, information_id: str) -> Optional[Information]:
        """
        Retrieve a document by ID.

        Parameters
        ----------
        information_id: str
            The ID of the document to retrieve.

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
        Perform a similarity search. Return the informations most similar informations to the query.

        Parameters
        ----------
        query: Union[str, List[float]]
            The query string or embedding vector.
        k: int
            The maximum number of results to return.
        threshold: float
            The similarity threshold. If the similarity score is below this value,
            the information will not be included in the results.

        Returns
        -------
        List[Information]
            The most similar informations, ordered by similarity (highest first).
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
        Save the database state.
        After calling this method the vector database is guaranteed to be saved i.e. persisted.
        However, some implementations might keep the database persisted at all times making the use of this method superfluous.
        It is recommended to call this method after any operation that modifies the database to ensure the changes are saved when using any implementation.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close the database connection.

        This is optional and may not be needed for all implementations.
        """
        pass
