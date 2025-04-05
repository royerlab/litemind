import copy
import os
import pickle
import uuid
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from arbol import aprint
from pydantic import BaseModel
from scipy.spatial import cKDTree

from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.information.information_base import InformationBase
from litemind.agent.augmentations.vector_db.default_vector_db import (
    DefaultVectorDatabase,
)
from litemind.apis.base_api import BaseApi
from litemind.media.media_base import MediaBase


class InMemoryVectorDatabase(DefaultVectorDatabase):
    """
    An in-memory vector database that also serves as an augmentation.

    This implementation uses a KD-tree for efficient similarity searches.
    It's useful for quick prototyping and testing with better performance
    than a naive implementation.

    If a location is provided, the database will be loaded from disk if it exists,
    otherwise a new database will be created and can be persisted with save().
    """

    def __init__(
        self,
        name: str = "InMemoryVectorDB",
        description: Optional[str] = None,
        embedding_function: Optional[
            Callable[[List[Information]], List[List[float]]]
        ] = None,
        embedding_length: Optional[int] = None,
        location: Optional[str] = None,
        api: BaseApi = None,
    ):
        """
        Initialize the vector database.

        Parameters
        ----------
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        embedding_function: Optional[Callable[[List[Information]], List[List[float]]]]
            A function that takes a list of informations and returns a list of embedding vectors.
            If not provided, a default embedding function will be used.
        embedding_length: Optional[int]
            The length of the embedding vectors.
        location: Optional[str]
            Path where to store the database. If provided and the database exists,
            it will be loaded from this location.
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

        if api is None:
            from litemind import CombinedApi

            self.api = CombinedApi()
        else:
            self.api = api

        # Store location for saving
        self.location = location

        # Initialize informations dict:
        self.informations: Dict[str, Information] = {}

        # Initialize embedding vectors:
        self.embedding_vectors = None

        # Initialize the kd tree:
        self.kdtree = None  # Will be initialized when informations are added
        self.id_to_index: Dict[str, int] = (
            {}
        )  # Maps information IDs to their index in the KDTree
        self.index_to_id: Dict[int, str] = {}  # Maps KDTree indices to information IDs
        self.rebuild_needed = False  # Flag to enable lazy rebuilding

        # Try to load existing database if location is provided
        if location and self._database_exists(location):
            self._load()
            return

    def _database_exists(self, location: str) -> bool:
        """Check if database files exist at the specified location."""
        metadata_path = os.path.join(location, "metadata.pkl")
        vectors_path = os.path.join(location, "vectors.npy")
        return os.path.exists(metadata_path) and os.path.exists(vectors_path)

    def _load(self) -> None:
        """Load the database from disk."""
        try:
            # Ensure directory exists
            if not os.path.exists(self.location):
                return

            # Load metadata
            metadata_path = os.path.join(self.location, "metadata.pkl")
            if not os.path.exists(metadata_path):
                return

            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            self.informations = metadata["informations"]
            self.id_to_index = metadata["id_to_index"]
            self.index_to_id = metadata["index_to_id"]
            self.embedding_length = metadata["embedding_length"]

            # Load vectors
            vectors_path = os.path.join(self.location, "vectors.npy")
            if os.path.exists(vectors_path):
                self.embedding_vectors = np.load(vectors_path)

            # Mark that we need to rebuild the KD-tree
            self.rebuild_needed = True
            self._rebuild_kdtree(force=True)

            aprint(
                f"Loaded vector database from {self.location} with {len(self.informations)} informations"
            )
        except Exception as e:
            aprint(f"Error loading database: {e}")
            # Initialize empty database if load fails
            self.informations = {}
            self.embedding_vectors = None
            self.id_to_index = {}
            self.index_to_id = {}
            self.rebuild_needed = False

    def _rebuild_kdtree(self, force=False):
        """
        Rebuild the KD-tree with current embeddings.

        Parameters
        ----------
        force: bool
            If True, forces rebuilding even if not flagged as needed.
        """
        if not force and not self.rebuild_needed:
            return

        if self.embedding_vectors is None or len(self.informations) == 0:
            self.kdtree = None
            self.id_to_index = {}
            self.index_to_id = {}
            return

        # Build the KD-tree
        self.kdtree = cKDTree(self.embedding_vectors)
        self.rebuild_needed = False

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
        if not informations:
            return []

        added_ids = []

        # Generate IDs for informations that don't have them
        for info in informations:
            if info.id is None:
                info.set_id(str(uuid.uuid4()))
            added_ids.append(info.id)

        # If embeddings aren't provided, compute them:
        if embeddings is None:
            aprint(f"Computing embeddings for {len(informations)} informations")
            embeddings = self.embedding_function(informations)

            # Set the embeddings in the informations
            for i, info in enumerate(informations):
                info.embedding = embeddings[i]

        # Ensure embeddings are a numpy array:
        embeddings = np.array(embeddings)

        # Normalize embeddings for faster cosine similarity later
        embeddings = embeddings / np.maximum(
            np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-10
        )

        # Initialize if this is the first addition
        if self.embedding_vectors is None:
            self.embedding_vectors = embeddings
            # Set up index mappings
            for i, info_id in enumerate(added_ids):
                self.informations[info_id] = informations[i]
                self.id_to_index[info_id] = i
                self.index_to_id[i] = info_id
        else:
            # Update the existing vectors
            current_size = len(self.informations)
            # Create mappings for new informations
            for i, info_id in enumerate(added_ids):
                self.informations[info_id] = informations[i]
                self.id_to_index[info_id] = current_size + i
                self.index_to_id[current_size + i] = info_id

            # Append to the embedding array
            self.embedding_vectors = np.vstack([self.embedding_vectors, embeddings])

        self.rebuild_needed = True

        return added_ids

    def get_information(self, information_id: str) -> Optional[InformationBase]:
        """
        Retrieve an information by ID.

        Parameters
        ----------
        information_id: str
            The ID of the information to retrieve.

        Returns
        -------
        Optional[InformationBase]
            The information, or None if not found.
        """
        return self.informations.get(information_id)

    def delete(self, ids: List[str]) -> None:
        """
        Delete informations from the database.

        Parameters
        ----------
        ids: List[str]
            The IDs of the informations to delete.
        """
        if not ids:
            return

        # Identify which informations to keep
        indices_to_delete = []
        for info_id in ids:
            if info_id in self.id_to_index:
                indices_to_delete.append(self.id_to_index[info_id])
                del self.informations[info_id]
                del self.id_to_index[info_id]

        if indices_to_delete and self.embedding_vectors is not None:
            # Create a mask of indices to keep
            keep_mask = np.ones(len(self.embedding_vectors), dtype=bool)
            keep_mask[indices_to_delete] = False

            # Filter embeddings
            self.embedding_vectors = self.embedding_vectors[keep_mask]

            # Rebuild mappings
            self.index_to_id = {}
            self.id_to_index = {}
            for i, info_id in enumerate(self.informations.keys()):
                self.id_to_index[info_id] = i
                self.index_to_id[i] = info_id

            self.rebuild_needed = True

    def clear(self) -> None:
        """Clear all informations from the database."""
        self.informations.clear()
        self.embedding_vectors = None
        self.kdtree = None
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.rebuild_needed = False

    def similarity_search(
        self,
        query: Union[str, BaseModel, MediaBase, Information, List[float]],
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[Information]:
        """
        Perform similarity search using the KD-tree.

        Parameters
        ----------
        query: Union[str, BaseModel, MediaBase, Information, List[float]]
            The query string, Pydantic base model, media object, information object, or embedding vector.
        k: int
            The maximum number of informations to return.
        threshold: float
            The score threshold_dict for the augmentation. If the score is below this value, the augmentation will not be used.

        Returns
        -------
        List[Information]
            The most similar informations, ordered by similarity (highest first).
        """

        # If the query is empty, return an empty list:
        if not self.informations:
            return []

        # If no documents are requested we obey:
        if k <= 0:
            return []

        # Ensure KD-tree is built if needed
        self._rebuild_kdtree()

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

        # Make sure we don't request more neighbors than we have informations
        k = min(k, len(self.informations))

        # If there's no KD-tree (no informations), return empty list
        if self.kdtree is None:
            return []

        # Query the KD-tree for nearest neighbors
        distances, indices = self.kdtree.query(query_vector, k=k)

        # Convert results to information list
        results = []

        # Handle both k_dict=1 and k_dict>1 cases by ensuring indices is always iterable
        if k == 1:
            # For k_dict=1, convert numpy scalar to Python int
            indices = [int(indices)]
            distances = [float(distances)]
        else:
            # For k_dict>1, convert each numpy int to Python int
            indices = [int(idx) for idx in indices]
            distances = [float(dist) for dist in distances]

        for i, idx in enumerate(indices):
            info_id = self.index_to_id[idx]
            info: Information = copy.copy(self.informations[info_id])

            # Calculate similarity score from distance
            similarity = 1.0 - (distances[i] * distances[i] / 2.0)
            info.score = similarity

            if similarity >= threshold:
                # Set the metadata for the information
                results.append(info)

        # Sort by similarity score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    # Implement AugmentationBase methods
    def get_relevant_informations(
        self,
        query: Union[str, BaseModel, MediaBase, Information],
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[InformationBase]:
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
        """Save the database to disk if a location was provided."""
        if not self.location:
            aprint("No location provided, can't save the database")
            return

        try:
            # Ensure directory exists
            os.makedirs(self.location, exist_ok=True)

            # Save metadata
            metadata = {
                "informations": self.informations,
                "id_to_index": self.id_to_index,
                "index_to_id": self.index_to_id,
                "embedding_length": self.embedding_length,
            }

            with open(os.path.join(self.location, "metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Save vectors if they exist
            if self.embedding_vectors is not None:
                np.save(
                    os.path.join(self.location, "vectors.npy"), self.embedding_vectors
                )

            aprint(
                f"Saved vector database to {self.location} with {len(self.informations)} informations"
            )
        except Exception as e:
            aprint(f"Error saving database: {e}")
            import traceback

            traceback.print_stack()

    def close(self) -> None:
        """Close the database"""
        self.save()
