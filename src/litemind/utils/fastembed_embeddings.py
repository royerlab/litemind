"""Text embedding generation using the FastEmbed library with dimensionality reduction."""

from functools import lru_cache
from typing import List, Optional, Sequence

from arbol import aprint

from litemind.utils.random_projector import DeterministicRandomProjector


@lru_cache()
def is_fastembed_available() -> bool:
    """
    Check that the fastembed library is installed and working.

    Actually imports fastembed to verify it works, since version
    mismatches in transitive dependencies (like tokenizers) can cause
    import errors even when the module is installed.

    Returns
    -------
    bool
        True if fastembed can be successfully imported.
    """
    try:
        from fastembed import TextEmbedding  # noqa: F401

        return True
    except Exception:
        return False


def fastembed_text(
    texts: List[str],
    model_name: Optional[str] = "BAAI/bge-large-en-v1.5",
    dimensions: int = 512,
    **kwargs,
) -> Sequence[Sequence[float]]:
    """
    Generate text embeddings using the FastEmbed library.

    Embeds the input texts using the specified model, then reduces the
    dimensionality via a deterministic random projection.

    Parameters
    ----------
    texts : list of str
        The texts to embed.
    model_name : str, optional
        The FastEmbed model name. Default is ``"BAAI/bge-large-en-v1.5"``.
    dimensions : int, optional
        The target embedding dimension after projection. Default is 512.
    **kwargs
        Additional keyword arguments passed to
        ``DeterministicRandomProjector``.

    Returns
    -------
    list of list of float
        The projected embeddings, one per input text.

    Raises
    ------
    ValueError
        If the specified model is not supported by FastEmbed.
    """
    from fastembed import TextEmbedding

    # Check that the fastembed library is installed:
    supported_models = TextEmbedding.list_supported_models()

    # Check if the fastembed library is installed:
    supported_models = [model["model"] for model in supported_models]

    # Check if model is supported:
    if model_name not in supported_models:
        for model in supported_models:
            aprint(model)
        raise ValueError(
            f"Model {model_name} is not supported. Supported models are: {supported_models}"
        )

    # Create a TextEmbedding object:
    model = TextEmbedding(model_name=model_name)

    # Embed the texts:
    embeddings = list(model.embed(texts))

    # Create a DeterministicRandomProjector object:
    drp = DeterministicRandomProjector(
        original_dim=len(embeddings[0]), reduced_dim=dimensions, **kwargs
    )

    # Project the embeddings:
    embeddings = drp.transform(embeddings)

    # Make sure that the embeddings are a list of lists by conversion:
    embeddings = [list(embedding) for embedding in embeddings]

    # Return the embeddings:
    return embeddings
