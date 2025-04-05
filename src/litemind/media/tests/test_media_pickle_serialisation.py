import pandas as pd
from pydantic import BaseModel

from litemind.media.types.media_code import Code
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_object import Object
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video


def test_text_pickle_serialization():
    instance = Text(text="Some text", extra_data=123)

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Text.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Text.from_base64(base64_data)
    assert reloaded_base64 == instance


def test_image_pickle_serialization():
    instance = Image(
        uri="file://path/to/image.png", attributes={"width": 800, "height": 600}
    )

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Image.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Image.from_base64(base64_data)
    assert reloaded_base64 == instance


def test_video_pickle_serialization():
    instance = Video(uri="file://path/to/video.mp4", duration=120)

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Video.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Video.from_base64(base64_data)
    assert reloaded_base64 == instance


def test_code_pickle_serialization():
    instance = Code(code="print('Hello world')", lang="python")

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Code.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Code.from_base64(base64_data)
    assert reloaded_base64 == instance


def test_document_pickle_serialization():
    instance = Document(uri="file://path/to/document.pdf", title="My Document")

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Document.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Document.from_base64(base64_data)
    assert reloaded_base64 == instance


def test_json_pickle_serialization():
    instance = Json(json={"key": "value", "nested": [1, 2, 3]})

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Json.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Json.from_base64(base64_data)
    assert reloaded_base64 == instance


# Define CustomObject at module level to avoid pickling issues
class CustomObject(BaseModel):
    type: str
    data: int


def test_object_pickle_serialization():
    instance = Object(
        object_=CustomObject(type="example", data=42), attributes={"key": "value"}
    )

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Object.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Object.from_base64(base64_data)
    assert reloaded_base64 == instance


def test_table_pickle_serialization():
    # Create a DataFrame
    df = pd.DataFrame({"Col1": ["A", "C"], "Col2": ["B", "D"]})

    instance = Table.from_dataframe(df)

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Table.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Table.from_base64(base64_data)
    assert reloaded_base64 == instance


def test_complex_data_pickle_serialization():
    """Test with more complex nested data structures."""
    complex_instance = Json(
        json={
            "text": Text(text="Nested text"),
            "code": Code(code="print('Hello')", lang="python"),
            "numbers": [1, 2, 3, 4, 5],
            "nested_dict": {"key1": "value1", "key2": 42, "key3": [True, False, None]},
        }
    )

    # Test pickle serialization
    pickled_data = complex_instance.to_pickle()
    reloaded_pickle = Json.from_pickle(pickled_data)
    assert reloaded_pickle == complex_instance

    # Test base64 serialization
    base64_data = complex_instance.to_base64()
    reloaded_base64 = Json.from_base64(base64_data)
    assert reloaded_base64 == complex_instance


def test_large_data_pickle_serialization():
    """Test with large data to ensure serialization works for large objects."""
    large_text = "x" * 100000  # 100KB of data
    instance = Text(text=large_text)

    # Test pickle serialization
    pickled_data = instance.to_pickle()
    reloaded_pickle = Text.from_pickle(pickled_data)
    assert reloaded_pickle == instance

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Text.from_base64(base64_data)
    assert reloaded_base64 == instance


def test_circular_reference_pickle_serialization():
    """Test serialization with circular references."""
    # Create objects with circular references
    obj1 = {}
    obj2 = {"ref_to_obj1": obj1}
    obj1["ref_to_obj2"] = obj2

    instance = Json(json=obj1)

    # Test pickle serialization (pickle handles circular references)
    pickled_data = instance.to_pickle()
    reloaded_pickle = Json.from_pickle(pickled_data)

    # Check that the circular reference is preserved
    assert reloaded_pickle.json["ref_to_obj2"]["ref_to_obj1"] is reloaded_pickle.json

    # Test base64 serialization
    base64_data = instance.to_base64()
    reloaded_base64 = Json.from_base64(base64_data)

    # Check that the circular reference is preserved
    assert reloaded_base64.json["ref_to_obj2"]["ref_to_obj1"] is reloaded_base64.json
