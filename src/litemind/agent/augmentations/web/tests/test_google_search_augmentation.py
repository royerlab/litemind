import os
import pytest
from litemind.agent.augmentations.web.google_search_augmentation import GoogleSearchAugmentation
from litemind.agent.augmentations.augmentation_base import Document

@pytest.fixture
def google_search_augmentation():
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    return GoogleSearchAugmentation(api_key=api_key, cse_id=cse_id)

def test_initialization_with_env_vars():
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    augmentation = GoogleSearchAugmentation()
    assert augmentation.api_key == api_key
    assert augmentation.cse_id == cse_id

def test_initialization_with_params():
    api_key = "test_api_key"
    cse_id = "test_cse_id"
    augmentation = GoogleSearchAugmentation(api_key=api_key, cse_id=cse_id)
    assert augmentation.api_key == api_key
    assert augmentation.cse_id == cse_id

def test_initialization_missing_api_key():
    with pytest.raises(ValueError, match="Google API key not found"):
        GoogleSearchAugmentation(api_key=None, cse_id="test_cse_id")


def test_get_relevant_documents_with_string_query(google_search_augmentation):
    query = "Python programming"
    documents = google_search_augmentation.get_relevant_documents(query=query, k=3)
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert "Title" in doc.content
        assert "URL" in doc.content
        assert "Snippet" in doc.content

def test_get_relevant_documents_with_document_query(google_search_augmentation):
    query = Document(content="Python programming")
    documents = google_search_augmentation.get_relevant_documents(query=query, k=3)
    assert len(documents) > 0
    for doc in documents:
        assert isinstance(doc, Document)
        assert "Title" in doc.content
        assert "URL" in doc.content
        assert "Snippet" in doc.content

def test_get_relevant_documents_invalid_query(google_search_augmentation):
    query = ""
    documents = google_search_augmentation.get_relevant_documents(query=query, k=3)
    assert len(documents) == 0