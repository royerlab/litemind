import os
import tempfile
import uuid

import pytest

from litemind import VECDB_IMPLEMENTATIONS
from litemind.agent.augmentations.augmentation_base import AugmentationBase
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.tests.utils.hash_embeddings import (
    simple_hash_embeddings,
)
from litemind.media.types.media_text import Text


def _get_temp_folder():
    """Create a temporary folder for vector database files"""
    return os.path.join(tempfile.gettempdir(), "augmentation_test_" + str(uuid.uuid4()))


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_augmentation_init(vector_db_api):
    """Test initialization with name and description (AugmentationBase behavior)"""
    # Create database with custom name and description
    db = vector_db_api(
        name="test_augmentation",
        description="Test description",
        location=_get_temp_folder(),
    )

    # Verify it adheres to AugmentationBase interface
    assert isinstance(db, AugmentationBase)
    assert db.name == "test_augmentation"
    assert db.description == "Test description"

    # Test with default description
    db = vector_db_api(name="test_augmentation2", location=_get_temp_folder())
    assert db.description == "test_augmentation2 Augmentation"


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_get_relevant_documents(vector_db_api):
    """Test the get_relevant_documents method from AugmentationBase"""
    # Create and prepare database
    db = vector_db_api(
        location=_get_temp_folder(),
        embedding_function=lambda texts: simple_hash_embeddings(texts, dim=128),
    )

    # Add test informations
    docs = [
        Information(Text("First test document"), metadata={"index": 1}),
        Information(Text("Second test document"), metadata={"index": 2}),
        Information(
            Text("Third document with different content"), metadata={"index": 3}
        ),
    ]
    db.add_informations(docs)

    # Test with string query
    results = db.get_relevant_informations("test document", k=2)

    # Verify results conform to AugmentationBase expectations
    assert isinstance(results, list)
    assert 1 <= len(results) <= 2
    assert all(isinstance(doc, Information) for doc in results)

    # Test with Document query
    doc_query = Information(Text("test document query"))
    results = db.get_relevant_informations(doc_query, k=2)

    assert isinstance(results, list)
    assert len(results) <= 2
    assert all(isinstance(doc, Information) for doc in results)


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_get_relevant_documents_iterator(vector_db_api):
    """Test the get_relevant_documents_iterator method from AugmentationBase"""
    # Create and prepare database
    db = vector_db_api(
        location=_get_temp_folder(),
        embedding_function=lambda texts: simple_hash_embeddings(texts, dim=128),
    )

    # Add test informations
    docs = [
        Information(Text("First test document"), metadata={"index": 1}),
        Information(Text("Second test document"), metadata={"index": 2}),
        Information(
            Text("Third document with different content"), metadata={"index": 3}
        ),
    ]
    db.add_informations(docs)

    # Test the iterator method
    iterator = db.get_relevant_informations_iterator("test query", k=2)

    # Verify it's an iterator
    assert hasattr(iterator, "__iter__")
    assert hasattr(iterator, "__next__")

    # Convert to list and check contents
    results = list(iterator)
    assert len(results) <= 2
    assert all(isinstance(doc, Information) for doc in results)


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_string_representations(vector_db_api):
    """Test the string representation methods from AugmentationBase"""
    db = vector_db_api(name="test_augmentation", location=_get_temp_folder())

    # Test string representations
    assert str(db) == "test_augmentation"

    # repr should contain class name and name attribute
    assert "test_augmentation" in repr(db)
    assert vector_db_api.__name__ in repr(db)
