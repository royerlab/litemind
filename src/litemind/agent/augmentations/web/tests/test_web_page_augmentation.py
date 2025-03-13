import pytest

from litemind.agent.augmentations.augmentation_base import Document
from litemind.agent.augmentations.web.web_page_augmentation import WebPageAugmentation


class MockVectorDatabase:
    def __init__(self):
        self.documents = []

    def add_documents(self, docs):
        self.documents.extend(docs)

    def get_relevant_documents(self, query, k=5):
        return self.documents[:k]

@pytest.fixture
def vector_database():
    return MockVectorDatabase()

@pytest.fixture
def web_page_augmentation(vector_database):
    return WebPageAugmentation(vector_database=vector_database)

def test_initialization_with_urls(vector_database):
    urls = ["http://example.com"]
    augmentation = WebPageAugmentation(vector_database=vector_database, urls=urls)
    assert len(augmentation.loaded_urls) == 1
    assert "http://example.com" in augmentation.loaded_urls

def test_add_url_success(web_page_augmentation):
    url = "http://example.com"
    success = web_page_augmentation.add_url(url)
    assert success
    assert url in web_page_augmentation.loaded_urls

def test_add_url_already_loaded(web_page_augmentation):
    url = "http://example.com"
    web_page_augmentation.loaded_urls.add(url)
    success = web_page_augmentation.add_url(url)
    assert success
    assert len(web_page_augmentation.loaded_urls) == 1

def test_add_url_failure(web_page_augmentation):
    url = "http://invalid-url"
    success = web_page_augmentation.add_url(url)
    assert not success
    assert url not in web_page_augmentation.loaded_urls

def test_get_relevant_documents_with_string_query(web_page_augmentation):
    url = "http://example.com"
    web_page_augmentation.add_url(url)
    query = "example"
    documents = web_page_augmentation.get_relevant_documents(query=query, k=1)
    assert len(documents) == 1
    assert isinstance(documents[0], Document)

def test_get_relevant_documents_with_document_query(web_page_augmentation):
    url = "http://example.com"
    web_page_augmentation.add_url(url)
    query = Document(content="example")
    documents = web_page_augmentation.get_relevant_documents(query=query, k=1)
    assert len(documents) == 1
    assert isinstance(documents[0], Document)