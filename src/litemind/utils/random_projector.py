from typing import List, Sequence, Union

import numpy as np


class DeterministicRandomProjector:
    """
    Dimensionality reduction via a fixed random projection matrix.

    Uses a Gaussian random matrix to project high-dimensional vectors
    into a lower-dimensional space. The projection is deterministic
    when a fixed random seed is provided.

    Parameters
    ----------
    original_dim : int
        The dimensionality of the input vectors.
    reduced_dim : int
        The target dimensionality after projection.
    random_state : int, optional
        Random seed for reproducibility. Default is 42.
    """

    def __init__(self, original_dim: int, reduced_dim: int, random_state: int = 42):
        self.original_dim = original_dim
        self.reduced_dim = reduced_dim
        rng = (
            np.random.default_rng(random_state)
            if random_state is not None
            else np.random.default_rng()
        )
        self.R = rng.standard_normal((reduced_dim, original_dim)) / np.sqrt(
            original_dim
        )

    def transform(
        self, embeddings: Union[Sequence[Sequence[float]], np.ndarray]
    ) -> List[List[float]]:
        """
        Project a batch of vectors onto the reduced dimension.

        Parameters
        ----------
        embeddings : Union[Sequence[Sequence[float]], np.ndarray]
            The input vectors to be projected.

        Returns
        -------
        list of list of float
            The projected vectors in the reduced dimension.
        """

        # Apply matrix multiplication for each embedding
        reduced = [(self.R @ np.array(e)).tolist() for e in embeddings]

        return reduced
