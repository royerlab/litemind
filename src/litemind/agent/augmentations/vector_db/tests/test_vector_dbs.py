import random
import string
import time
import uuid

import pytest

from litemind import VECDB_IMPLEMENTATIONS
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.tests.utils.hash_embeddings import (
    simple_hash_embeddings,
)
from litemind.media.types.media_text import Text


def _get_temp_folder():
    import os
    import tempfile

    return os.path.join(tempfile.gettempdir(), "vector_db_test" + str(uuid.uuid4()))


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_add_documents(vector_db_api):
    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Information(Text("Hello world"), id=None, metadata={"author": "Alice"})
    doc2 = Information(Text("Goodbye world"), id=None, metadata={"author": "Bob"})

    # Add the document to the database:
    doc_ids = vector_db.add_informations([doc1, doc2])

    # Check that the document was added:
    assert len(doc_ids) == 2
    assert doc1.id in doc_ids
    assert doc2.id in doc_ids


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_get_document(vector_db_api):
    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc = Information(Text("Hello world"), metadata={"author": "Alice"})

    # Add the document to the database:
    vector_db.add_informations([doc])

    # Retrieve the document:
    retrieved_doc = vector_db.get_information(doc.id)

    # Check that the document was retrieved:
    assert retrieved_doc is not None
    assert retrieved_doc.content == "Hello world"
    assert retrieved_doc.metadata == {"author": "Alice"}


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_similarity_search(vector_db_api):
    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Information(Text("Hello world"), id=None, metadata={"author": "Alice"})
    doc2 = Information(Text("Goodbye world"), id=None, metadata={"author": "Bob"})

    # Add the document to the database:
    vector_db.add_informations([doc1, doc2])

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
    doc1 = Information(Text("Hello world"), metadata={"author": "Alice"})
    doc2 = Information(Text("Goodbye world"), metadata={"author": "Bob"})

    # Add the document to the database:
    vector_db.add_informations([doc1, doc2])

    # Delete the document:
    vector_db.delete([doc1.id])

    # Check that the document was deleted:
    assert vector_db.get_information(doc1.id) is None
    assert vector_db.get_information(doc2.id) is not None


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_clear(vector_db_api):
    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Information(Text("Hello world"), metadata={"author": "Alice"})
    doc2 = Information(Text("Goodbye world"), metadata={"author": "Bob"})

    # Add the document to the database:
    vector_db.add_informations([doc1, doc2])

    # Clear the database:
    vector_db.clear()

    # Check that the informations were deleted:
    assert vector_db.get_information(doc1.id) is None
    assert vector_db.get_information(doc2.id) is None


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_get_relevant_documents(vector_db_api):
    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Create a new document:
    doc1 = Information(Text("Hello world"), id=None, metadata={"author": "Alice"})
    doc2 = Information(Text("Goodbye world"), id=None, metadata={"author": "Bob"})

    # Add the document to the database:
    vector_db.add_informations([doc1, doc2])

    # Perform a similarity search for "Hello"
    results = vector_db.get_relevant_informations("Hello", k=1)

    # Check that the correct document was retrieved:
    assert len(results) == 1
    assert results[0].content == "Hello world"


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_complex_similarity_search(vector_db_api):
    # Create a new document:
    vector_db = vector_db_api(location=_get_temp_folder())

    # Add some informations to the database
    documents = [
        Information(Text("first content"), id=None, metadata={"key": "value1"}),
        Information(Text("second content"), id=None, metadata={"key": "value2"}),
        Information(Text("third content"), id=None, metadata={"key": "value3"}),
        Information(Text("fourth content"), id=None, metadata={"key": "value4"}),
    ]

    # Add the informations to the database
    vector_db.add_informations(documents)

    # Perform a similarity search for "third content"
    results = vector_db.similarity_search("content #3", k=1)

    # Check that the correct document was retrieved:
    assert len(results) == 1
    assert results[0].content == "third content"
    assert results[0].metadata == {"key": "value3"}


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_vector_db_stress_test(vector_db_api):
    """
    Stress test for vector databases - measures performance with a large number of informations.
    """
    # Configuration
    num_documents = 5000
    doc_length = 100
    num_queries = 10
    k_retrievals = 5

    def embedding_function(text: str):
        return simple_hash_embeddings(text, dim=256)

    # Create a new vector database with our fast hash embeddings
    vector_db = vector_db_api(
        location=_get_temp_folder(), embedding_function=embedding_function
    )

    # Generate random informations
    documents = []
    print(f"\nGenerating {num_documents} random informations for {vector_db.name}...")

    for i in range(num_documents):
        # Generate random content with some words to query later
        content = " ".join(random.choices(string.ascii_lowercase, k=doc_length))
        special_term = (
            f"unique_term_{i % 100}"  # Add trackable terms to some informations
        )
        if i % 3 == 0:
            content = content + " " + special_term

        doc = Information(
            id=None,
            media=Text(content),
            metadata={"index": i, "special_term": special_term if i % 3 == 0 else None},
        )
        documents.append(doc)

    # Time document insertion
    start_time = time.time()
    doc_ids = vector_db.add_informations(documents)
    insertion_time = time.time() - start_time

    assert len(doc_ids) == num_documents

    # Doing one query in case the database needs to be initialized / updated:
    vector_db.similarity_search("unique_term_0", k=k_retrievals)

    # Time document retrieval with random queries
    total_query_time = 0
    for i in range(num_queries):
        # For some queries, use terms we know exist
        if i % 2 == 0:
            query = f"unique_term_{random.randint(0, 99)}"
        else:
            # Random query
            query = " ".join(random.choices(string.ascii_lowercase, k=5))

        start_time = time.time()
        results = vector_db.similarity_search(query, k=k_retrievals)
        query_time = time.time() - start_time
        total_query_time += query_time

        #

        assert len(results) <= k_retrievals

    avg_query_time = total_query_time / num_queries

    # Output performance metrics
    print(f"\nPerformance metrics for {vector_db.name}:")
    print(f"- Documents: {num_documents}")
    print(
        f"- Insertion time: {insertion_time:.4f} seconds ({num_documents / insertion_time:.1f} docs/sec)"
    )
    print(f"- Average query time: {avg_query_time:.4f} seconds")
    print(
        f"Note: this benchmark does not include the computation of a complex embedding, we are using a simple hash embedding here."
    )

    # Return metrics for potential further analysis
    return {
        "db_type": vector_db.name,
        "num_documents": num_documents,
        "insertion_time": insertion_time,
        "insertion_rate": num_documents / insertion_time,
        "avg_query_time": avg_query_time,
    }


@pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
def test_persistence(vector_db_api):
    """Test that vector databases can save and load their state."""
    temp_dir = _get_temp_folder()

    # Create test informations
    docs = [
        Information(
            Text("This is the first test document"), id=None, metadata={"index": 1}
        ),
        Information(
            Text("Another document for testing"), id=None, metadata={"index": 2}
        ),
        Information(
            Text("Third document with different content"),
            id=None,
            metadata={"index": 3},
        ),
    ]

    # Create database and add informations
    db1 = vector_db_api(location=temp_dir)
    db1.add_informations(docs)

    # Perform a search to verify it works
    results1 = db1.similarity_search("test document", k=2)
    assert len(results1) == 2

    # Save the database
    db1.save()

    # Close the database
    db1.close()

    # Delete the original database instance
    del db1

    # Create a new database instance pointing to the same location
    db2 = vector_db_api(location=temp_dir)

    # Verify search still works after loading
    results2 = db2.similarity_search("test document", k=2)
    assert len(results2) == 2

    # Verify search results are consistent (comparing content)
    assert {d.content for d in results1} == {d.content for d in results2}

    # Test modifying and re-saving
    new_doc = Information(
        Text("Fourth document added after loading"), id=None, metadata={"index": 4}
    )
    db2.add_informations([new_doc])

    # Save the modified database
    db2.save()

    # Close the database
    db2.close()

    # Delete the original database instance
    del db2

    # Load again and verify the new document is searchable
    db3 = vector_db_api(location=temp_dir)
    results3 = db3.similarity_search("fourth document", k=1)
    assert len(results3) == 1
    assert "Fourth document" in results3[0].content
