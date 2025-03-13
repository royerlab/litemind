import os
import uuid
from typing import Callable, List, Optional, Union, Sequence

from arbol import aprint

from litemind.agent.augmentations.document import Document
from litemind.apis.base_api import BaseApi
from litemind.rag.vector_db.default_vector_db import DefaultVectorDatabase


class QdrantVectorDatabase(DefaultVectorDatabase):
    """
    A vector database implementation using Qdrant.
    
    This implementation uses the Qdrant vector database to store and retrieve documents.
    It requires the qdrant-client package to be installed.
    """
    
    def __init__(
        self,
        name: str = "QdrantVectorDB",
        description: Optional[str] = None,
        collection_name: str = "documents",
        embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
        embedding_length: int = 768,
        url: Optional[str] = None,
        location: Optional[str] = None,
        qdrant_api_key: Optional[str] = None,
        api: BaseApi = None,
    ):
        """
        Initialize the Qdrant vector database.
        
        Parameters
        ----------
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        collection_name: str
            The name of the collection in Qdrant.
        embedding_function: Optional[Callable]
            A function that takes a string and returns an embedding vector.
        embedding_length: int
            The length of the embedding vectors.
        url: Optional[str]
            The URL of the Qdrant server. If None, will use a local instance.
        location: Optional[str]
            The location of the local Qdrant instance. Only used if url is None.
        qdrant_api_key: Optional[str]
            The API key for the Qdrant server.
        api: BaseApi
            The API to use for embedding. If not provided, uses litemind's default API.

        """
        super().__init__(name=name, description=description)

        if api is None:
            from litemind import CombinedApi
            self.api = CombinedApi()
        else:
            self.api = api

        self.collection_name = collection_name
        self.embedding_function = embedding_function or self._default_embedding_function
        self.embedding_length = embedding_length
        
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http import models
            
            if url:
                self.client = QdrantClient(url=url, api_key=qdrant_api_key)
            else:
                if location is None:
                    # Get the user's home folder:
                    home_folder = os.path.expanduser("~")

                    # Qdrant data folder:
                    location = os.path.join(home_folder, ".qdrant_data")

                    # Create the Qdrant data folder:
                    os.makedirs(location, exist_ok=True)

                # Use the local Qdrant instance at the specified location:
                self.client = QdrantClient(path=location)


            # Create collection if it doesn't exist
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if collection_name not in collection_names:
                aprint(f"Creating new Qdrant collection: {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_length,
                        distance=models.Distance.COSINE,
                    ),
                )
                
        except ImportError:
            raise ImportError("Qdrant not installed. Please install it with 'pip install qdrant-client'.")
    
    def _default_embedding_function(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """
        Default embedding function using litemind's API.
        
        Parameters
        ----------
        texts: str
            List of texts (strings) Tto embed.
            
        Returns
        -------
        List[List[float]]
            The embedding vectors.
        """
        # Embed the texts:
        embeddings = self.api.embed_texts(texts=texts,
                                          dimensions=self.embedding_length)

        # Return the embeddings:
        return embeddings
    
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
        from qdrant_client.http import models
        
        # Generate IDs for documents without IDs
        for doc in documents:
            if doc.id is None:
                doc.id = str(uuid.uuid4())
        
        # Create embeddings if not provided
        if embeddings is None:
            aprint(f"Computing embeddings for {len(documents)} documents")
            embeddings = self.embedding_function([doc.content for doc in documents])
        
        # Prepare points for Qdrant
        points = []
        for doc, emb in zip(documents, embeddings):
            # Create a point
            payload = {
                "content": doc.content,
                "metadata": doc.metadata or {},
            }
            points.append(models.PointStruct(
                id=doc.id,
                vector=emb,
                payload=payload,
            ))
        
        # Upsert points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )
        
        return [doc.id for doc in documents]
    
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
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[document_id],
        )
        
        if not points:
            return None
            
        point = points[0]
        return Document(
            id=str(point.id),
            content=point.payload["content"],
            metadata=point.payload["metadata"],
        )
    
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
            embeddings = list(self.embedding_function([query]))
            query_vector = embeddings[0]
        else:
            query_vector = query
        
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
        )
        
        documents = []
        for result in search_result:
            doc = Document(
                id=str(result.id),
                content=result.payload["content"],
                metadata=result.payload["metadata"],
                score=result.score,
            )
            documents.append(doc)
            
        return documents
    
    def delete(self, ids: List[str]) -> None:
        """
        Delete documents from the database.
        
        Parameters
        ----------
        ids: List[str]
            The IDs of the documents to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )
    
    def clear(self) -> None:
        """Clear all documents from the database."""
        try:
            # Delete the collection
            self.client.delete_collection(collection_name=self.collection_name)
            
            # Recreate the collection
            from qdrant_client.http import models
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_length,
                    distance=models.Distance.COSINE,
                ),
            )
        except Exception as e:
            # If collection doesn't exist, that's fine
            aprint(f"Error clearing collection: {e}")
    
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
