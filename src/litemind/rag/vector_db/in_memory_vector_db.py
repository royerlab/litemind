import copy
import uuid
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from arbol import aprint

from litemind.agent.augmentations.augmentation_base import Document
from litemind.rag.vector_db.default_vector_db import DefaultVectorDatabase


class InMemoryVectorDatabase(DefaultVectorDatabase):
    """
    An in-memory vector database that also serves as an augmentation.
    
    This is a simple implementation that stores documents and their embeddings in memory.
    It's useful for quick prototyping and testing, but not recommended for production
    use with large document collections.
    """
    
    def __init__(
        self,
        name: str = "InMemoryVectorDB",
        description: Optional[str] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        location: Optional[str] = None,
    ):
        """
        Initialize the vector database.
        
        Parameters
        ----------
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        embedding_function: Optional[Callable[[str], List[float]]]
            A function that takes a string and returns an embedding vector.
            If not provided, a default embedding function will be used.
        """
        super().__init__(name=name, description=description)
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, List[float]] = {}
        self.embedding_function = embedding_function or self._default_embedding_function

    
    def _default_embedding_function(self, text: str) -> List[float]:
        """
        Default embedding function using litemind's API.
        
        Parameters
        ----------
        text: str
            The text to embed.
            
        Returns
        -------
        List[float]
            The embedding vector.
        """
        from litemind import CombinedApi
        api = CombinedApi()
        embeddings = api.embed_texts([text])
        return embeddings[0]
    
    def add_documents(self, documents: List[Document], embeddings: Optional[List[List[float]]] = None) -> List[str]:
        """
        Add documents to the database.
        
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
        added_ids = []
        
        # If embeddings aren't provided, compute them
        if embeddings is None:
            aprint(f"Computing embeddings for {len(documents)} documents")
            embeddings = [self.embedding_function(doc.content) for doc in documents]
        
        for doc, emb in zip(documents, embeddings):
            # Generate an ID if none exists
            if doc.id is None:
                doc.id = str(uuid.uuid4())
                
            self.documents[doc.id] = doc
            self.embeddings[doc.id] = emb
            added_ids.append(doc.id)
            
        return added_ids
    
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
        return self.documents.get(document_id)
    
    def similarity_search(self, query: Union[str, List[float]], k: int = 5) -> List[Document]:
        """
        Perform similarity search.
        
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
        if isinstance(query, str):
            query_vector = self.embedding_function(query)
        else:
            query_vector = query
            
        # Compute similarities
        similarities = {}
        for doc_id, embedding in self.embeddings.items():
            # Cosine similarity
            dot_product = np.dot(query_vector, embedding)
            query_norm = np.linalg.norm(query_vector)
            emb_norm = np.linalg.norm(embedding)
            
            if query_norm == 0 or emb_norm == 0:
                similarity = 0
            else:
                similarity = dot_product / (query_norm * emb_norm)
            
            similarities[doc_id] = similarity
            
        # Sort by similarity
        sorted_ids = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)
        
        # Return top k documents with scores
        results = []
        for doc_id in sorted_ids[:k]:
            doc = copy.copy(self.documents[doc_id])
            doc.score = similarities[doc_id]
            results.append(doc)
            
        return results
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the database.
        
        Parameters
        ----------
        ids: List[str]
            The IDs of the documents to delete.
        """
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
            if doc_id in self.embeddings:
                del self.embeddings[doc_id]
    
    def clear(self) -> None:
        """Clear all documents from the database."""
        self.documents.clear()
        self.embeddings.clear()
    
    # Implement AugmentationBase methods
    def get_relevant_documents(self, query: Union[str, Document], k: int = 5) -> List[Document]:
        """
        Get relevant documents for a query.
        
        Parameters
        ----------
        query: Union[str, Document]
            The query to retrieve documents for.
        k: int
            The maximum number of documents to retrieve.
            
        Returns
        -------
        List[Document]
            A list of relevant documents.
        """
        if isinstance(query, Document):
            query = query.content
        return self.similarity_search(query, k)
