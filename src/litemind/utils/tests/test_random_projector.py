import numpy as np
import pytest

from litemind.utils.random_projector import DeterministicRandomProjector


def test_random_projector_initialization():
    original_dim = 768
    reduced_dim = 128
    random_state = 42

    projector = DeterministicRandomProjector(original_dim, reduced_dim, random_state)

    assert projector.R.shape == (
        reduced_dim,
        original_dim,
    ), "Projection matrix shape mismatch"
    assert np.allclose(
        np.mean(projector.R), 0, atol=1e-1
    ), "Projection matrix mean is not close to 0"
    assert np.allclose(
        np.std(projector.R), 1.0 / np.sqrt(original_dim), atol=1e-1
    ), "Projection matrix std deviation is incorrect"


def test_random_projector_transform():
    original_dim = 768
    reduced_dim = 128
    random_state = 42

    projector = DeterministicRandomProjector(original_dim, reduced_dim, random_state)

    batch_embeddings = [[0.1] * original_dim, [0.2] * original_dim]
    batch_reduced = projector.transform(batch_embeddings)

    assert len(batch_reduced) == 2, "Batch transformation output size mismatch"
    assert (
        len(batch_reduced[0]) == reduced_dim
    ), "Reduced dimension length mismatch for batch transformation"
    assert (
        len(batch_reduced[1]) == reduced_dim
    ), "Reduced dimension length mismatch for batch transformation"


if __name__ == "__main__":
    pytest.main()
