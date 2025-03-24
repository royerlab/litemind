import pytest
from arbol import aprint
import os
import chromadb 

# Import needed Object/Function
from litemind.agent.rag.chromadb_embeddings import EmbeddingModel
from litemind.agent.rag.vector_database import ChromaDB
from litemind.apis.model_features import ModelFeatures
from litemind import OpenAIApi, GeminiApi, OllamaApi, AnthropicApi


@pytest.mark.parametrize("api_class", [OpenAIApi])#OpenAIApi,GeminiApi,OllamaApi,AnthropicApi
def test_create_chroma_client(api_class):

    # Create OpenAI API
    api = api_class()

     # check at least one model is available for Text Embedding, if not, skip test:
    if not api.has_model_support_for(features=ModelFeatures.TextEmbeddings):
        aprint(f"Skipping test for {api_class.__name__} as no text model is available.")
        return
    
    # Create embedding function
    embedding_function = EmbeddingModel(api=api, model_name="text-embedding-3-small")

    # Create chroma client
    chroma_client = ChromaDB(persistent_path=os.getcwd(), collection_name="my_test_collection", embedding_function=embedding_function)

    # Create the collection
    chroma_client.create_collection()


    assert chroma_client.get_collection_name() == "my_test_collection"

    assert chroma_client.persistent_path == os.getcwd()

    assert isinstance(chroma_client.embedding_function, EmbeddingModel)

    assert isinstance(chroma_client.get_collection("my_test_collection"), chromadb.Collection)

    assert isinstance(chroma_client.get_settings(), chromadb.config.Settings)



@pytest.mark.parametrize("api_class", [OpenAIApi])
def test_rename_chroma_client(api_class):

    # Create OpenAI API
    api = api_class()

     # check at least one model is available for Text Embedding, if not, skip test:
    if not api.has_model_support_for(features=ModelFeatures.TextEmbeddings):
        aprint(f"Skipping test for {api_class.__name__} as no text model is available.")
        return
    
    # Create embedding function
    embedding_function = EmbeddingModel(api=api, model_name="text-embedding-3-small")

    # Create chroma client
    chroma_client = ChromaDB(persistent_path=os.getcwd(), collection_name="my_test_collection_2", embedding_function=embedding_function)

    # Create the collection
    chroma_client.create_collection()

    # Modify name
    chroma_client.rename_collection(name="my_test_collection_3")

    assert chroma_client.get_collection_name() == "my_test_collection_3"


@pytest.mark.parametrize("api_class", [OpenAIApi])
def test_delete_chroma_client(api_class):
    
    # Create OpenAI API
    api = api_class()

     # check at least one model is available for Text Embedding, if not, skip test:
    if not api.has_model_support_for(features=ModelFeatures.TextEmbeddings):
        aprint(f"Skipping test for {api_class.__name__} as no text model is available.")
        return
    
    # Create embedding function
    embedding_function = EmbeddingModel(api=api, model_name="text-embedding-3-small")

    # Create chroma client
    chroma_client = ChromaDB(persistent_path=os.getcwd(), collection_name="my_test_collection_4", embedding_function=embedding_function)

    # Create the collection
    chroma_client.create_collection()

    # delete collection
    chroma_client.delete_collection(name="my_test_collection_4")

    with pytest.raises(RuntimeError) as e_info:
        chroma_client.get_collection("my_test_collection_4")
        
        raise RuntimeError("Collection my_test_collection_4 does not exist.")

    assert e_info.value.args[0] == " Collection my_test_collection_4 does not exist."

    



