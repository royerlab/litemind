import os
import uuid
from typing import Callable, List, Optional, Union

import numpy as np
from arbol import aprint
from pydantic import BaseModel

from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.default_vector_db import (
    DefaultVectorDatabase,
)
from litemind.apis.base_api import BaseApi
from litemind.media.media_base import MediaBase


class QdrantVectorDatabase(DefaultVectorDatabase):
    """
    A vector database implementation using Qdrant.

    This implementation uses the Qdrant vector database to store and retrieve informations.
    It requires the qdrant-client package to be installed.
    """

    def __init__(
        self,
        name: str = "QdrantVectorDB",
        description: Optional[str] = None,
        collection_name: str = "informations",
        embedding_function: Optional[
            Callable[[List[Information]], List[List[float]]]
        ] = None,
        embedding_length: Optional[int] = None,
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
        embedding_function: Optional[Callable[[List[Information]], List[List[float]]]]
            A function that takes a list of informations and returns a list of embedding vectors.
            If not provided, a default embedding function will be used.
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
        super().__init__(
            name=name,
            description=description,
            embedding_function=embedding_function,
            embedding_length=embedding_length,
            api=api,
        )

        self.collection_name = collection_name

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
                        size=self.embedding_length,
                        distance=models.Distance.COSINE,
                    ),
                )

        except ImportError:
            raise ImportError(
                "Qdrant not installed. Please install it with 'pip install qdrant-client'."
            )

    def add_informations(
        self,
        informations: List[Information],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        Add informations to the database.

        Parameters
        ----------
        informations: List[Information]
            The informations to add to the database.
        embeddings: Optional[List[List[float]]]
            The embeddings for the informations. If not provided, they will be computed.

        Returns
        -------
        List[str]
            The IDs of the added informations.
        """
        from qdrant_client.http import models

        # Generate IDs for informations without IDs
        for info in informations:
            if info.id is None:
                info.set_id(str(uuid.uuid4()))

        # Create embeddings if not provided
        if embeddings is None:
            aprint(f"Computing embeddings for {len(informations)} informations")
            embeddings = self.embedding_function(informations)

        # Prepare points for Qdrant
        points = []
        for info, emb in zip(informations, embeddings):
            # Create a point
            payload = {"info": info.to_base64()}
            points.append(
                models.PointStruct(
                    id=info.id,
                    vector=emb,
                    payload=payload,
                )
            )

        # Upsert points to Qdrant
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

        return [info.id for info in informations]

    def get_information(self, information_id: str) -> Optional[Information]:
        """
        Retrieve a information by ID.

        Parameters
        ----------
        information_id: str
            The ID of the information to retrieve.

        Returns
        -------
        Optional[Information]
            The information, or None if not found.
        """
        points = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[information_id],
        )

        if not points:
            return None

        point = points[0]
        return Information.from_base64(point.payload["info"])

    def similarity_search(
        self,
        query: Union[str, BaseModel, MediaBase, Information, List[float]],
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Information]:
        """
        Perform similarity search.

        Parameters
        ----------
        query: Union[str, BaseModel, MediaBase, Information, List[float]]
            The query string or embedding vector.
        k: int
            The maximum number of results to return.
        threshold: float
            The similarity threshold_dict. If the similarity is below this value, the result will not be used.

        Returns
        -------
        List[Information]
            The most similar informations, ordered by similarity (highest first).
        """

        # If no documents are requested we obey:
        if k <= 0:
            return []

        if isinstance(query, list) and isinstance(query[0], float):

            # If the query is already a vector, we don't need to normalize it:
            query_vector = query

        else:
            # Normalize the query
            query = self._normalize_query(query)

            # Wrap in Information:
            query = Information(query)

            # Get the embedding for the query:
            query_vector = list(self.embedding_function([query])[0])

        # Convert to numpy array:
        query_vector = np.array(query_vector)

        # Normalize query vector
        query_norm = np.linalg.norm(query_vector)
        if query_norm > 0:
            query_vector = query_vector / query_norm

        # Perform similarity search
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=k,
        )

        results = []
        for result in search_result.points:

            # Deserialize the information from the payload
            info = Information.from_base64(result.payload["info"])
            # Information.from_json(result.payload["info"])

            # Set the score (similarity) for the information
            info.score = result.score

            # Check if the score is above the threshold_dict
            if info.score > threshold:
                # Add the information to the list
                results.append(info)

        return results

    def delete(self, ids: List[str]) -> None:
        """
        Delete informations from the database.

        Parameters
        ----------
        ids: List[str]
            The IDs of the informations to delete.
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )

    def clear(self) -> None:
        """Clear all informations from the database."""
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
    def get_relevant_informations(
        self,
        query: Union[str, BaseModel, MediaBase, Information],
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Information]:
        """
        Get relevant informations for a query.

        Parameters
        ----------
        query: Union[str, BaseModel, MediaBase, Information]
            The query to retrieve informations for.
        k: int
            The maximum number of informations to retrieve.
        threshold: float
            The score threshold_dict for the augmentation. If the score is below this value, the augmentation will not be used.

        Returns
        -------
        List[Information]
            A list of relevant informations.
        """

        # Normalize the query
        query = self._normalize_query(query)

        # Perform similarity search
        return self.similarity_search(query, k, threshold)

    def save(self) -> None:
        # No need to implement save for Qdrant as it stays in sync with the database.
        pass

    def close(self) -> None:
        """Close the connection to the Qdrant database."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()
