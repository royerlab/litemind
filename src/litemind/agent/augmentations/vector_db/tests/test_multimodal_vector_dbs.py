import hashlib
import os
import tempfile
import uuid
from typing import List, Union

import pytest

from litemind import VECDB_IMPLEMENTATIONS
from litemind.agent.augmentations.information.information import Information
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_code import Code
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.ressources.media_resources import MediaResources


def _get_temp_folder():
    """Create a temporary folder for vector database files"""
    return os.path.join(
        tempfile.gettempdir(), "multimodal_vecdb_test_" + str(uuid.uuid4())
    )


def multimodal_hash_embeddings(
    content: Union[str, bytes, List[str], List[bytes]], dim: int = 128
) -> List[List[float]]:
    """
    A simple embedding function that works with different data types:
    - For strings: hash the string
    - For bytes: hash the bytes
    - For URIs: load the content and hash it

    Returns a list of embeddings, each with dimension 'dim'
    """
    if not isinstance(content, list):
        content = [content]

    embeddings = []
    for item in content:
        if isinstance(item, str):
            if item.startswith("file://"):
                # Handle file URI by reading the file and hashing its content
                file_path = item[7:]
                try:
                    with open(file_path, "rb") as f:
                        data = f.read()
                    hash_obj = hashlib.sha256(data)
                except:
                    # If file can't be opened, hash the URI string
                    hash_obj = hashlib.sha256(item.encode())
            else:
                # Regular string
                hash_obj = hashlib.sha256(item.encode())
        else:
            # For bytes or other binary data
            hash_obj = hashlib.sha256(
                item if isinstance(item, bytes) else str(item).encode()
            )

        # Generate a deterministic embedding from the hash
        hash_digest = hash_obj.digest()
        # Create a fixed-length embedding by cycling through the hash
        embedding = []
        for i in range(dim):
            val = hash_digest[i % len(hash_digest)] / 255.0  # Normalize to [0, 1]
            # Make it span [-1, 1] with some minor variations for better vector space
            embedding.append((val * 2 - 1) * (0.8 + 0.2 * (i % 7) / 7))

        embeddings.append(embedding)

    return embeddings


class TestMultimodalVectorDB(MediaResources):
    """Test suite for multimodal vector database functionality"""

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_text_informations(self, vector_db_api):
        """Test adding and retrieving text informations"""
        # Create vector database with multimodal embeddings
        db = vector_db_api(
            location=_get_temp_folder(),
            embedding_function=lambda texts: multimodal_hash_embeddings(texts, dim=128),
        )

        # Add text informations
        infos = [
            Information(
                Text("Python is a programming language"),
                metadata={"index": 1},
            ),
            Information(
                Text("Machine learning algorithms process data"),
                metadata={"index": 2},
            ),
            Information(
                Text("Vector databases store and retrieve embeddings"),
                metadata={"index": 3},
            ),
        ]
        db.add_informations(infos)

        # Test retrieval
        results = db.get_relevant_informations("Python programming", k=2)

        assert len(results) > 0
        assert all(isinstance(info, Information) for info in results)
        # assert any("Python" in info.content for info in results)

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_json_informations(self, vector_db_api):
        """Test adding and retrieving JSON informations"""
        db = vector_db_api(
            location=_get_temp_folder(),
            embedding_function=lambda texts: multimodal_hash_embeddings(texts, dim=128),
        )

        # Add JSON informations
        infos = [
            Information(
                Json.from_string('{"name": "John", "age": 30, "city": "New York"}'),
                metadata={"index": 1},
            ),
            Information(
                Json.from_string('{"name": "Alice", "age": 25, "city": "Boston"}'),
                metadata={"index": 2},
            ),
            Information(
                Json.from_string('{"name": "Bob", "age": 35, "city": "Chicago"}'),
                metadata={"index": 3},
            ),
        ]
        db.add_informations(infos)

        # Test retrieval
        results = db.get_relevant_informations('{"city": "New York"}', k=1)

        assert len(results) > 0
        assert all(isinstance(info, Information) for info in results)
        # assert any("New York" in doc.content for doc in results)

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_code_informations(self, vector_db_api):
        """Test adding and retrieving code informations"""
        db = vector_db_api(
            location=_get_temp_folder(),
            embedding_function=lambda texts: multimodal_hash_embeddings(texts, dim=128),
        )

        # Add code informations
        infos = [
            Information(
                Code("def hello():\n    print('Hello, world!')", lang="python"),
                metadata={"language": "python", "index": 1},
            ),
            Information(
                Code(
                    "function factorial(n) {\n    return n <= 1 ? 1 : n * factorial(n-1);\n}",
                    lang="python",
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
        # assert any("factorial" in info.media for info in results)

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_image_informations(self, vector_db_api):
        """Test adding and retrieving image informations"""
        db = vector_db_api(
            location=_get_temp_folder(),
            embedding_function=lambda content: multimodal_hash_embeddings(
                content, dim=128
            ),
        )

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
        assert any("cat" in info.content for info in results)

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_audio_informations(self, vector_db_api):
        """Test adding and retrieving audio informations"""
        db = vector_db_api(
            location=_get_temp_folder(),
            embedding_function=lambda content: multimodal_hash_embeddings(
                content, dim=128
            ),
        )

        # Add audio informations
        harvard_audio_uri = self.get_local_test_audio_uri("harvard.wav")

        infos = [
            Information(
                Audio(harvard_audio_uri),
                metadata={"speech": "Harvard sentences", "index": 1},
            )
        ]
        db.add_informations(infos)

        # Test retrieval with URI
        results = db.get_relevant_informations(harvard_audio_uri, k=1)

        assert len(results) > 0
        assert all(isinstance(info, Information) for info in results)
        assert any("harvard" in info.content.lower() for info in results)

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_video_informations(self, vector_db_api):
        """Test adding and retrieving video informations"""
        db = vector_db_api(
            location=_get_temp_folder(),
            embedding_function=lambda content: multimodal_hash_embeddings(
                content, dim=128
            ),
        )

        # Add video informations
        flying_video_uri = self.get_local_test_video_uri("flying.mp4")

        infos = [
            Information(
                Video(flying_video_uri),
                metadata={"subject": "flying object", "index": 1},
            )
        ]
        db.add_informations(infos)

        # Test retrieval with URI
        results = db.get_relevant_informations(flying_video_uri, k=1)

        assert len(results) > 0
        assert all(isinstance(info, Information) for info in results)
        assert any("flying" in info.content.lower() for info in results)

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_table_informations(self, vector_db_api):
        """Test adding and retrieving table informations"""
        db = vector_db_api(
            location=_get_temp_folder(),
            embedding_function=lambda content: multimodal_hash_embeddings(
                content, dim=128
            ),
        )

        # Simulate a table as CSV content
        table1 = "name,age,city\nJohn,30,New York\nAlice,25,Boston\nBob,35,Chicago"
        table2 = (
            "product,price,quantity\nApple,1.20,100\nBanana,0.50,150\nOrange,0.80,75"
        )

        infos = [
            Information(
                Table.from_csv_string(table1),
                metadata={"type": "people", "index": 1},
            ),
            Information(
                Table.from_csv_string(table2),
                metadata={"type": "products", "index": 2},
            ),
        ]
        db.add_informations(infos)

        # Test retrieval
        results = db.get_relevant_informations("name,age", k=1)

        assert len(results) > 0
        assert all(isinstance(info, Information) for info in results)
        # assert any("John" in doc.content for doc in results)

    @pytest.mark.parametrize("vector_db_api", VECDB_IMPLEMENTATIONS)
    def test_document_informations(self, vector_db_api):
        """Test adding and retrieving mixed document types"""
        db = vector_db_api(
            location=_get_temp_folder(),
            embedding_function=lambda content: multimodal_hash_embeddings(
                content, dim=128
            ),
        )

        # Add informations of different types
        python_img_uri = self.get_local_test_image_uri("python.png")
        harvard_audio_uri = self.get_local_test_audio_uri("harvard.wav")

        infos = [
            Information(
                Text("Python is a programming language"),
                metadata={"index": 1},
            ),
            Information(
                Image(python_img_uri),
                metadata={"subject": "logo", "index": 2},
            ),
            Information(
                Audio(harvard_audio_uri),
                metadata={"speech": "Harvard sentences", "index": 3},
            ),
            Information(
                Code(
                    "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
                    lang="python",
                ),
                metadata={"language": "python", "index": 4},
            ),
        ]
        db.add_informations(infos)

        # Test retrieval of specific types
        results = db.get_relevant_informations("Python programming", k=2)
        assert len(results) > 0
        # assert any("Python" in info.content for info in results)

        # Test retrieval with image URI
        image_results = db.get_relevant_informations(python_img_uri, k=1)
        assert len(image_results) > 0
        # assert any("python" in info.content.lower() for info in image_results)
