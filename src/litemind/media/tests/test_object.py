import pytest
from pydantic import BaseModel

from litemind.media.types.media_json import Json
from litemind.media.types.media_object import Object
from litemind.media.types.media_text import Text


class ExampleModel(BaseModel):
    name: str
    value: int


def test_object_initialization():
    """Test that the Object is correctly initialized."""
    example_object = ExampleModel(name="Test", value=1)
    obj = Object(example_object)
    assert obj.object == example_object


def test_object_initialization_with_kwargs():
    """Test that the Object is correctly initialized with kwargs."""
    example_object = ExampleModel(name="Test", value=1)
    obj = Object(example_object, some_kwarg="test")
    assert obj.object == example_object
    assert obj.attributes == {"some_kwarg": "test"}


def test_object_initialization_none_object():
    """Test that the Object raises ValueError when object is None."""
    with pytest.raises(ValueError, match="Object cannot be None"):
        Object(None)


def test_object_initialization_invalid_object():
    """Test that the Object raises ValueError when object is not a Pydantic BaseModel."""
    with pytest.raises(ValueError, match="Object must be a Pydantic BaseModel"):
        Object("not a BaseModel")


def test_to_json_string():
    """Test that the object is correctly converted to a JSON string."""
    example_object = ExampleModel(name="Test", value=1)
    obj = Object(example_object)
    json_string = obj.to_json_string()
    assert isinstance(json_string, str)
    assert '"name":"Test"' in json_string
    assert '"value":1' in json_string


def test_to_json_media():
    """Test that the object is correctly converted to a Json media."""
    example_object = ExampleModel(name="Test", value=1)
    obj = Object(example_object)
    json_media = obj.to_json_media()
    assert isinstance(json_media, Json)
    assert (
        "name" in obj.to_json_string()
        and "Test" in obj.to_json_string()
        and "value" in obj.to_json_string()
        and "1" in obj.to_json_string()
    )


def test_to_markdown_string():
    """Test that the object is correctly converted to a markdown string."""
    example_object = ExampleModel(name="Test", value=1)
    obj = Object(example_object)
    markdown_string = obj.to_markdown_string()
    assert isinstance(markdown_string, str)
    assert "```json" in markdown_string
    assert '"name":"Test"' in markdown_string
    assert '"value":1' in markdown_string


def test_to_markdown_text_media():
    """Test that the object is correctly converted to a markdown Text media."""
    example_object = ExampleModel(name="Test", value=1)
    obj = Object(example_object)
    markdown_text_media = obj.to_markdown_text_media()
    assert isinstance(markdown_text_media, Text)
    assert "```json" in markdown_text_media.text
    assert '"name":"Test"' in markdown_text_media.text
    assert '"value":1' in markdown_text_media.text


def test_object_str_representation():
    """Test the string representation of the Object."""
    example_object = ExampleModel(name="Test", value=1)
    obj = Object(example_object)
    assert str(obj) == obj.to_json_string()
