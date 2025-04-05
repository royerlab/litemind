import os
import tempfile
import uuid

import pytest

from litemind import VECDB_IMPLEMENTATIONS
from litemind.agent.augmentations.information.information import Information
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_code import Code
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.ressources.media_resources import MediaResources


def _get_temp_folder():
    """Create a temporary folder for vector database files"""
    return os.path.join(tempfile.gettempdir(), "augmentation_test_" + str(uuid.uuid4()))


class TestMultimodalRelevantInformations(MediaResources):
    """Test suite for get_relevant_informations with multimodal content"""

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_multimodal_basic(self, vector_db_api):
        """Test get_relevant_informations with various media types"""
        db = vector_db_api(location=_get_temp_folder())

        # Get real media URIs
        python_img_uri = self.get_local_test_image_uri("python.png")
        cat_img_uri = self.get_local_test_image_uri("cat.jpg")
        harvard_audio_uri = self.get_local_test_audio_uri("harvard.wav")
        flying_video_uri = self.get_local_test_video_uri("flying.mp4")

        # Create table data
        people_table = Table.from_csv_string(
            "name,age,city\nJohn,30,New York\nAlice,25,Boston\nBob,35,Chicago"
        )

        # Create document data
        pdf_doc_uri = self.get_local_test_document_uri("intracktive_preprint.pdf")
        doc = Document(pdf_doc_uri)

        # Create test data with different media types
        test_data = [
            Information(Text("Python programming guide"), metadata={"type": "text"}),
            Information(
                Code("def hello():\n    print('Hello world!')", lang="python"),
                metadata={"type": "code"},
            ),
            Information(
                Image(python_img_uri), metadata={"type": "image", "subject": "logo"}
            ),
            Information(
                Image(cat_img_uri), metadata={"type": "image", "subject": "animal"}
            ),
            Information(
                Audio(harvard_audio_uri), metadata={"speech": "Harvard sentences"}
            ),
            Information(Json({"name": "test", "value": 42}), metadata={"type": "json"}),
            Information(Video(flying_video_uri), metadata={"subject": "flying object"}),
            Information(people_table, metadata={"type": "people"}),
            Information(doc, metadata={"type": "document"}),
        ]

        # Add all informations to the DB
        db.add_informations(test_data)

        # Test 1: Text query
        text_results = db.get_relevant_informations("Python programming", k=3)
        assert len(text_results) > 0
        assert all(isinstance(doc, Information) for doc in text_results)

        # Test 2: Information with Text query
        info_text_query = Information(Text("Python programming"))
        info_text_results = db.get_relevant_informations(info_text_query, k=3)
        assert len(info_text_results) > 0

        # Test 3: Code query
        code_results = db.get_relevant_informations(
            Information(Code("print('Hello')", lang="python")), k=3
        )
        assert len(code_results) > 0

        # Test 4: Image query using real image URI
        image_results = db.get_relevant_informations(python_img_uri, k=3)
        assert len(image_results) > 0

        # Test 5: Audio query using real audio URI
        audio_results = db.get_relevant_informations(harvard_audio_uri, k=3)
        assert len(audio_results) > 0

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_code(self, vector_db_api):
        """Test get_relevant_informations with code in different languages"""
        db = vector_db_api(location=_get_temp_folder())

        # Add code informations
        infos = [
            Information(
                Code("def hello():\n    print('Hello, world!')", lang="python"),
                metadata={"language": "python", "index": 1},
            ),
            Information(
                Code(
                    "function factorial(n) {\n    return n <= 1 ? 1 : n * factorial(n-1);\n}",
                    lang="javascript",
                ),
                metadata={"language": "javascript", "index": 2},
            ),
            Information(
                Code("SELECT * FROM users WHERE age > 18;", lang="sql"),
                metadata={"language": "sql", "index": 3},
            ),
        ]
        db.add_informations(infos)

        # Test retrieval
        results = db.get_relevant_informations("function factorial", k=2)
        assert len(results) > 0
        assert all(isinstance(info, Information) for info in results)

        # Test language-specific retrieval
        sql_results = db.get_relevant_informations(
            Information(Code("SELECT", lang="sql")), k=2
        )
        assert len(sql_results) > 0

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_images(self, vector_db_api):
        """Test get_relevant_informations with images"""
        db = vector_db_api(location=_get_temp_folder())

        # Add image informations
        python_img_uri = self.get_local_test_image_uri("python.png")
        cat_img_uri = self.get_local_test_image_uri("cat.jpg")
        panda_img_uri = self.get_local_test_image_uri("panda.jpg")

        infos = [
            Information(
                Image(python_img_uri),
                metadata={"subject": "logo", "index": 1},
            ),
            Information(
                Image(cat_img_uri),
                metadata={"subject": "animal", "index": 2},
            ),
            Information(
                Image(panda_img_uri),
                metadata={"subject": "animal", "index": 3},
            ),
        ]
        db.add_informations(infos)

        # Test retrieval with URI
        results = db.get_relevant_informations(cat_img_uri, k=1)
        assert len(results) > 0
        assert all(isinstance(info, Information) for info in results)

        # Test retrieval with Image object
        image_results = db.get_relevant_informations(
            Information(Image(panda_img_uri)), k=2
        )
        assert len(image_results) > 0

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_audio_video(self, vector_db_api):
        """Test get_relevant_informations with audio and video"""
        db = vector_db_api(location=_get_temp_folder())

        # Add audio and video informations
        harvard_audio_uri = self.get_local_test_audio_uri("harvard.wav")
        flying_video_uri = self.get_local_test_video_uri("flying.mp4")

        infos = [
            Information(
                Audio(harvard_audio_uri),
                metadata={"speech": "Harvard sentences", "index": 1},
            ),
            Information(
                Video(flying_video_uri),
                metadata={"subject": "flying object", "index": 1},
            ),
        ]
        db.add_informations(infos)

        # Test retrieval with Audio object
        audio_results = db.get_relevant_informations(
            Information(Audio(harvard_audio_uri)), k=1
        )
        assert len(audio_results) > 0

        # Test retrieval with Video object
        video_results = db.get_relevant_informations(
            Information(Video(flying_video_uri)), k=1
        )
        assert len(video_results) > 0

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_table(self, vector_db_api):
        """Test get_relevant_informations with tables"""
        db = vector_db_api(location=_get_temp_folder())

        # Create tables from CSV strings
        table1 = Table.from_csv_string(
            "name,age,city\nJohn,30,New York\nAlice,25,Boston\nBob,35,Chicago"
        )
        table2 = Table.from_csv_string(
            "product,price,quantity\nApple,1.20,100\nBanana,0.50,150\nOrange,0.80,75"
        )

        infos = [
            Information(table1, metadata={"type": "people", "index": 1}),
            Information(table2, metadata={"type": "products", "index": 2}),
        ]
        db.add_informations(infos)

        # Test retrieval with table query
        query_table = Table.from_csv_string("name,age\nJohn,30")
        table_results = db.get_relevant_informations(Information(query_table), k=1)
        assert len(table_results) > 0

        # Test retrieval with text query about table content
        text_results = db.get_relevant_informations("product prices", k=1)
        assert len(text_results) > 0

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_document(self, vector_db_api):
        """Test get_relevant_informations with documents"""
        db = vector_db_api(location=_get_temp_folder())

        # Get document URIs
        pdf_doc_uri = self.get_local_test_document_uri("sample.pdf")

        infos = [
            Information(
                Document(pdf_doc_uri), metadata={"type": "technical", "index": 1}
            )
        ]
        db.add_informations(infos)

        # Test retrieval with document query
        doc_results = db.get_relevant_informations(
            Information(Document(pdf_doc_uri)), k=1
        )
        assert len(doc_results) > 0

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_json(self, vector_db_api):
        """Test get_relevant_informations with JSON data"""
        db = vector_db_api(location=_get_temp_folder())

        # Create JSON objects
        json1 = Json({"weather": {"temperature": 25, "condition": "sunny"}})
        json2 = Json(
            {"user": {"name": "John", "age": 30, "roles": ["admin", "editor"]}}
        )

        infos = [
            Information(json1, metadata={"type": "weather", "index": 1}),
            Information(json2, metadata={"type": "user", "index": 2}),
        ]
        db.add_informations(infos)

        # Test retrieval with JSON query
        json_query = Json({"user": {"name": "John"}})
        json_results = db.get_relevant_informations(Information(json_query), k=1)
        assert len(json_results) > 0

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_mixed_media(self, vector_db_api):
        """Test retrieving mixed media types with the same query"""
        db = vector_db_api(location=_get_temp_folder())

        # Get media URIs
        python_img_uri = self.get_local_test_image_uri("python.png")
        harvard_audio_uri = self.get_local_test_audio_uri("harvard.wav")

        # Add informations of different types with related content
        infos = [
            Information(
                Text("Python programming language guide"), metadata={"topic": "python"}
            ),
            Information(
                Code("import numpy as np", lang="python"), metadata={"topic": "python"}
            ),
            Information(Image(python_img_uri), metadata={"topic": "python"}),
            Information(
                Text("Harvard sentences used for audio testing"),
                metadata={"topic": "audio"},
            ),
            Information(Audio(harvard_audio_uri), metadata={"topic": "audio"}),
            Information(
                Json({"language": "python", "version": "3.9"}),
                metadata={"topic": "python"},
            ),
        ]
        db.add_informations(infos)

        # Query with text and verify mixed results
        results = db.get_relevant_informations("Python programming", k=5)
        assert len(results) > 0

        # Check if we get different media types in results
        media_types = {type(result.content).__name__ for result in results}
        assert len(media_types) > 1

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_get_relevant_informations_edge_cases(self, vector_db_api):
        """Test edge cases for get_relevant_informations"""
        db = vector_db_api(location=_get_temp_folder())

        # Test with empty database
        empty_results = db.get_relevant_informations("test query", k=5)
        assert len(empty_results) == 0

        # Test with single item
        db.add_informations([Information(Text("Single test document"))])
        single_results = db.get_relevant_informations("test", k=1)
        assert len(single_results) == 1

        # Test with k=0 (should return empty list)
        zero_k_results = db.get_relevant_informations("test", k=0)
        assert len(zero_k_results) == 0

        # Test with threshold filtering
        threshold_results = db.get_relevant_informations("test", k=5, threshold=0.8)
        assert all(result.score >= 0.8 for result in threshold_results)

        # Test with duplicate content
        db.clear()
        db.add_informations(
            [
                Information(Text("Duplicate content")),
                Information(Text("Duplicate content")),
            ]
        )
        duplicate_results = db.get_relevant_informations("Duplicate", k=3)
        assert 1 <= len(duplicate_results) <= 2  # Should find at least one match
