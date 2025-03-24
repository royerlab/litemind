import pytest
from arbol import aprint

# Import needed Object/Function
from litemind.agent.rag.rag_agent import RagAgent
from litemind.agent.rag.chromadb_embeddings import EmbeddingModel
from litemind.agent.rag.vector_database import ChromaDB
from litemind import OpenAIApi
from litemind.apis.model_features import ModelFeatures


@pytest.mark.parametrize("api_class", [OpenAIApi])
def test_agent(api_class):

    # Create OpenAI API Object
    api = api_class()

    # check at least one model is available for Text Embedding, if not, skip test:
    if not api.has_model_support_for(features=ModelFeatures.TextEmbeddings):
        pytest.skip(f"Skipping test for {api_class.__name__} as no text model is available.")
    
    aprint('Model: text-embedding-3-small')
    # define path (Use the vectorestore created in advance)
    persistent_path = "C:\\Users\\dario\\OneDrive\\università\\MA\\Thesis\\litemind\\litemind"

    # Initialize Embedding model
    embedding_function = EmbeddingModel(api=api, model_name="text-embedding-3-small")

    # Initialize chroma client with created collection
    chroma_client = ChromaDB(persistent_path=persistent_path, collection_name="my_test_collection", embedding_function=embedding_function)

    # Initialize Rag Agent
    agent = RagAgent(api=api, chroma_vectorestore=chroma_client, model_name="gpt-4o-mini")

    response = agent("Make a summary about the article 'The hGIDGID4 E3 ubiquitin ligase complex targets'")


    assert response is not None

    assert len(agent.conversation) == 2

    assert agent.conversation[0].role == "user"

    assert ( 
        "Make a summary about the article 'The hGIDGID4 E3 ubiquitin ligase complex targets'" in agent.conversation[0] )
    
    assert agent.conversation[1].role == "assistant"

    assert len(response) == 1









    return None