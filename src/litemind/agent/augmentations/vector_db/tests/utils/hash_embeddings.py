import hashlib
from typing import List, Union

import numpy as np

from litemind.agent.augmentations.information.information import Information


def simple_hash_embeddings(
    informations: Union[Information, List[Information]], dim: int = 128
) -> np.ndarray:
    """
    Generate a simple deterministic embedding vector based on hashing.

    Parameters
    ----------
    informations: List[Document]
        The informations to create an embedding for
    dim: int
        Dimensionality of the embedding vector

    Returns
    -------
    np.ndarray
        A normalized embedding vector
    """
    if isinstance(informations, list):
        return np.array([simple_hash_embeddings(d, dim) for d in informations])

    # Convert Document to string:
    text = str(informations)

    # Create a hash of the text
    hasher = hashlib.sha256(text.encode("utf-8"))
    hash_bytes = hasher.digest()

    # Convert the hash to an array of floats
    # Reuse the hash bytes as needed to fill the dimensions
    values = []
    for i in range(dim):
        byte_idx = i % len(hash_bytes)
        values.append(float(hash_bytes[byte_idx]) / 255.0)

    # Convert to numpy array and normalize
    embedding = np.array(values)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding
