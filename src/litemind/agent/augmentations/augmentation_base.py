from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union

from pydantic import BaseModel

from litemind.agent.augmentations.document import Document


class AugmentationBase(ABC):
    """
    Abstract base class defining the interface for all augmentations.
    
    Augmentations provide additional context to the LLM by retrieving relevant
    documents based on a query.
    """
    
    def __init__(self, name: str, description: Optional[str] = None):
        """
        Create a new augmentation.
        
        Parameters
        ----------
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        """
        self.name = name
        self.description = description or f"{name} Augmentation"
    
    @abstractmethod
    def get_relevant_documents(self, query: Union[str, BaseModel], k: int = 5) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        
        Parameters
        ----------
        query: Union[str, BaseModel]
            The query to retrieve documents for. Can be a string or a structured Pydantic model.
        k: int
            The maximum number of documents to retrieve.
            
        Returns
        -------
        List[Document]
            A list of relevant documents.
        """
        pass
    
    def get_relevant_documents_iterator(self, query: Union[str, BaseModel], k: int = 5) -> Iterator[Document]:
        """
        Retrieve documents relevant to the query as an iterator.
        This default implementation calls get_relevant_documents and yields from the result.
        
        Parameters
        ----------
        query: Union[str, BaseModel]
            The query to retrieve documents for. Can be a string or a structured Pydantic model.
        k: int
            The maximum number of documents to retrieve.
            
        Returns
        -------
        Iterator[Document]
            An iterator over relevant documents.
        """
        for doc in self.get_relevant_documents(query, k=k):
            yield doc
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __str__(self):
        return self.name
