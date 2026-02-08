import hashlib
import shutil
import tempfile
from typing import List, Union

import pytest

from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import (
    InMemoryVectorDatabase,
)
from litemind.media.types.media_text import Text


def _simple_hash_embeddings(
    content: Union[str, bytes, List[str], List[bytes]], dim: int = 128
) -> List[List[float]]:
    """Hash-based embedding function for offline tests."""
    if not isinstance(content, list):
        content = [content]

    embeddings = []
    for item in content:
        if hasattr(item, "content"):
            text = str(item.content)
        elif isinstance(item, bytes):
            text = item.hex()
        else:
            text = str(item)

        # Trigram feature hashing for meaningful similarity
        embedding = [0.0] * dim
        text_lower = text.lower()
        for i in range(max(1, len(text_lower) - 2)):
            trigram = text_lower[i : i + 3]
            h = hashlib.sha256(trigram.encode()).digest()
            bucket = h[0] % dim
            embedding[bucket] += 1.0

        norm = sum(x * x for x in embedding) ** 0.5
        if norm > 0:
            embedding = [x / norm for x in embedding]
        else:
            h = hashlib.sha256(text.encode()).digest()
            for i in range(dim):
                embedding[i] = (h[i % len(h)] / 255.0) * 2 - 1
            norm = sum(x * x for x in embedding) ** 0.5
            if norm > 0:
                embedding = [x / norm for x in embedding]

        embeddings.append(embedding)
    return embeddings


class TestInMemoryVectorDatabase:
    @pytest.fixture
    def test_infos(self):
        """Create some test informations."""
        return [
            Information(Text("This is the first test information"), id="doc1"),
            Information(Text("Another information for testing"), id="doc2"),
            Information(Text("Third information with different content"), id="doc3"),
        ]

    @pytest.fixture
    def db_dir(self):
        """Create a temporary directory for the database."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_save_load(self, test_infos, db_dir):
        # Create a database and add informations
        db1 = InMemoryVectorDatabase(
            name="TestDB",
            location=db_dir,
            embedding_function=_simple_hash_embeddings,
            embedding_length=128,
        )
        db1.add_informations(test_infos)

        # Perform a search to verify it works
        results1 = db1.similarity_search("test information", k=2)
        assert len(results1) == 2
        assert results1[0].id in [doc.id for doc in test_infos]

        # Save the database
        db1.save()

        # Create a new database instance pointing to the same location
        db2 = InMemoryVectorDatabase(
            name="TestDB",
            location=db_dir,
            embedding_function=_simple_hash_embeddings,
            embedding_length=128,
        )

        # Verify informations were loaded
        assert len(db2.informations) == len(test_infos)
        for info_id in [info.id for info in test_infos]:
            assert info_id in db2.informations

        # Verify document content is preserved
        for info in test_infos:
            loaded_info = db2.get_information(info.id)
            assert loaded_info is not None
            assert loaded_info.media == info.media

        # Verify search still works after loading
        results2 = db2.similarity_search("test information", k=2)
        assert len(results2) == 2

        # Verify search results are consistent
        assert set(d.id for d in results1) == set(d.id for d in results2)

        # Test modifying and re-saving
        new_info = Information(
            Text("Fourth information added after loading"), id="info4"
        )
        db2.add_informations([new_info])
        db2.save()

        # Load again and verify the new document is there
        db3 = InMemoryVectorDatabase(
            name="TestDB",
            location=db_dir,
            embedding_function=_simple_hash_embeddings,
            embedding_length=128,
        )
        assert len(db3.informations) == len(test_infos) + 1
        assert "info4" in db3.informations
