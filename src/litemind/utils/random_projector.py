from typing import List, Sequence, Union

import numpy as np


class DeterministicRandomProjector:
    def __init__(self, original_dim: int, reduced_dim: int, random_state: int = 42):
        self.original_dim = original_dim
        self.reduced_dim = reduced_dim
        if random_state is not None:
            np.random.seed(random_state)
        self.R = np.random.randn(reduced_dim, original_dim) / np.sqrt(original_dim)

    def transform(
        self, embeddings: Union[Sequence[Sequence[float]], np.ndarray]
    ) -> List[List[float]]:
        """
        Project a batch of vectors onto the reduced dimension.

        Parameters
        ----------
        embeddings: Union[Sequence[Sequence[float]], np.ndarray]
            The input vectors to be projected.

        Returns
        -------
        List[List[float]]
            The projected vectors in the reduced dimension.
        """

        # Apply matrix multiplication for each embedding
        reduced = [(self.R @ np.array(e)).tolist() for e in embeddings]

        return reduced
