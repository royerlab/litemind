import pytest
from arbol import aprint
import numpy as np

# Import needed Object/Function
from litemind.agent.rag.chromadb_embeddings import EmbeddingModel
from litemind import OpenAIApi,GeminiApi,OllamaApi,AnthropicApi
from litemind.apis.model_features import ModelFeatures


@pytest.mark.parametrize("api_class", [OpenAIApi])#,GeminiApi,OllamaApi,AnthropicApi
def test_embedding_model(api_class):

    # Create OpenAI API
    api = api_class()

    # check at least one model is available for Text Embedding, if not, skip test:
    if not api.has_model_support_for(features=ModelFeatures.TextEmbeddings):
        aprint(f"Skipping test for {api_class.__name__} as no text model is available.")
        return
    
    aprint('Model: text-embedding-3-small')

    # Create embedding function
    embedding_function = EmbeddingModel(api=api, model_name="text-embedding-3-small")

    documet_chunks = [
        "This is the first chunks", 
        "I added also a second chunks",
        "this is your last chunks"
    ]

    # use embedding function on an example text
    embed_vectors = embedding_function(input=documet_chunks)

    # print vectors
    aprint(embed_vectors)

    # checks the len of the vector
    assert len(embed_vectors) == len(documet_chunks)

    # checks len of the vectors
    assert len(embed_vectors[0]) == 512

    # checks type of vectors
    assert embed_vectors[0][0].dtype == np.float32

