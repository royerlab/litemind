from abc import ABC, abstractmethod
from typing import List, Optional, Union

from litemind.agent.augmentations.augmentation_base import Document


class BaseVectorDatabase(ABC):
    """
    Abstract base class for vector databases.
    
    Vector databases store documents along with their vector embeddings to enable
    semantic similarity search.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """
        Add documents to the vector database.
        
        Parameters
        ----------
        documents: List[Document]
            The documents to add to the database.
        embeddings: Optional[List[List[float]]]
            The embeddings for the documents. If not provided, they will be computed.
            
        Returns
        -------
        List[str]
            The IDs of the added documents.
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a document by ID.
        
        Parameters
        ----------
        document_id: str
            The ID of the document to retrieve.
            
        Returns
        -------
        Optional[Document]
            The document, or None if not found.
        """
        pass
    
    @abstractmethod
    def similarity_search(self, query: Union[str, List[float]], k: int = 5) -> List[Document]:
        """
        Perform a similarity search.
        
        Parameters
        ----------
        query: Union[str, List[float]]
            The query string or embedding vector.
        k: int
            The maximum number of results to return.
            
        Returns
        -------
        List[Document]
            The most similar documents, ordered by similarity (highest first).
        """
        pass
    
    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the database.
        
        Parameters
        ----------
        ids: List[str]
            The IDs of the documents to delete.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the database."""
        pass
