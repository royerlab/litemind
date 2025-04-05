import shutil
import tempfile

import pytest

from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.vector_db.in_memory_vector_db import (
    InMemoryVectorDatabase,
)
from litemind.media.types.media_text import Text


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
        db1 = InMemoryVectorDatabase(name="TestDB", location=db_dir)
        db1.add_informations(test_infos)

        # Perform a search to verify it works
        results1 = db1.similarity_search("test information", k=2)
        assert len(results1) == 2
        assert results1[0].id in [doc.id for doc in test_infos]

        # Save the database
        db1.save()

        # Create a new database instance pointing to the same location
        db2 = InMemoryVectorDatabase(name="TestDB", location=db_dir)

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
        db3 = InMemoryVectorDatabase(name="TestDB", location=db_dir)
        assert len(db3.informations) == len(test_infos) + 1
        assert "info4" in db3.informations
