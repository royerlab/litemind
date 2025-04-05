from pprint import pprint

import pytest

from litemind.utils.fastembed_embeddings import fastembed_text, is_fastembed_available


@pytest.mark.skipif(
    not is_fastembed_available(), reason="fastembed library is not available"
)
def test_fastembed_text():
    texts = ["This is a tests sentence.", "Another tests sentence."]
    model_name = "BAAI/bge-small-en-v1.5"
    dimensions = 512

    embeddings = fastembed_text(texts, model_name=model_name, dimensions=dimensions)

    pprint(embeddings)

    assert len(embeddings) == len(
        texts
    ), "Number of embeddings should match number of input texts"
    assert all(
        len(embedding) == dimensions for embedding in embeddings
    ), "Each embedding should have the correct number of dimensions"
