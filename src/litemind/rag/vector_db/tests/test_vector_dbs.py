import uuid

import pytest

from litemind import VECDB_IMPLEMENTATIONS
from litemind.agent.augmentations.augmentation_base import Document

def _get_temp_folder():
    import tempfile
    import os
    return os.path.join(tempfile.gettempdir(), "vector_db_test"+str(uuid.uuid4()))

@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_add_documents(vector_db_api):

    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Document(id=None, content="Hello world", metadata={"author": "Alice"})
    doc2 = Document(id=None, content="Goodbye world", metadata={"author": "Bob"})

    # Add the document to the database:
    doc_ids = vector_db.add_documents([doc1, doc2])

    # Check that the document was added:
    assert len(doc_ids) == 2
    assert doc1.id in doc_ids
    assert doc2.id in doc_ids

@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_get_document(vector_db_api):

    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc = Document(content="Hello world", metadata={"author": "Alice"})

    # Add the document to the database:
    vector_db.add_documents([doc])

    # Retrieve the document:
    retrieved_doc = vector_db.get_document(doc.id)

    # Check that the document was retrieved:
    assert retrieved_doc is not None
    assert retrieved_doc.content == "Hello world"
    assert retrieved_doc.metadata == {"author": "Alice"}

@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_similarity_search(vector_db_api):

    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Document(id=None, content="Hello world", metadata={"author": "Alice"})
    doc2 = Document(id=None, content="Goodbye world", metadata={"author": "Bob"})

    # Add the document to the database:
    vector_db.add_documents([doc1, doc2])

    # Perform a similarity search for "Hello"
    results = vector_db.similarity_search("Hello", k=1)

    # Check that the correct document was retrieved:
    assert len(results) == 1
    assert results[0].content == "Hello world"

@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_delete(vector_db_api):

    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Document(content="Hello world", metadata={"author": "Alice"})
    doc2 = Document(content="Goodbye world", metadata={"author": "Bob"})

    # Add the document to the database:
    vector_db.add_documents([doc1, doc2])

    # Delete the document:
    vector_db.delete([doc1.id])

    # Check that the document was deleted:
    assert vector_db.get_document(doc1.id) is None
    assert vector_db.get_document(doc2.id) is not None

@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_clear(vector_db_api):

    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Document(content="Hello world", metadata={"author": "Alice"})
    doc2 = Document(content="Goodbye world", metadata={"author": "Bob"})

    # Add the document to the database:
    vector_db.add_documents([doc1, doc2])

    # Clear the database:
    vector_db.clear()

    # Check that the documents were deleted:
    assert vector_db.get_document(doc1.id) is None
    assert vector_db.get_document(doc2.id) is None

@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_get_relevant_documents(vector_db_api):

    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Document(id=None, content="Hello world", metadata={"author": "Alice"})
    doc2 = Document(id=None, content="Goodbye world", metadata={"author": "Bob"})

    # Add the document to the database:
    vector_db.add_documents([doc1, doc2])

    # Perform a similarity search for "Hello"
    results = vector_db.get_relevant_documents("Hello", k=1)

    # Check that the correct document was retrieved:
    assert len(results) == 1
    assert results[0].content == "Hello world"

@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_complex_similarity_search(vector_db_api):

    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Add some documents to the database
    documents = [
        Document(id=None, content="first content", metadata={"key": "value1"}),
        Document(id=None, content="second content", metadata={"key": "value2"}),
        Document(id=None, content="third content", metadata={"key": "value3"}),
        Document(id=None, content="fourth content", metadata={"key": "value4"})
    ]

    # Add the documents to the database
    vector_db.add_documents(documents)

    # Perform a similarity search for "third content"
    results = vector_db.similarity_search("content #3", k=1)

    # Check that the correct document was retrieved:
    assert len(results) == 1
    assert results[0].content == "third content"
    assert results[0].metadata == {"key": "value3"}