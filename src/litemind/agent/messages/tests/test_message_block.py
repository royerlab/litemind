import copy

from pandas import DataFrame
from pydantic import BaseModel

from litemind.agent.messages.message_block import MessageBlock
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_object import Object
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video


class ExampleModel(BaseModel):
    key: str
    value: str


def test_message_block_deepcopy():
    # Create a message block
    original_block = MessageBlock(media=Text("Sample content"))

    # Perform a deep copy of the message block
    copied_block = copy.deepcopy(original_block)

    # Verify that the copied block is equal to the original
    assert copied_block.get_content() == original_block.get_content()

    # Verify that the copied block is a different object
    assert copied_block is not original_block

    # if the content is a mutable object, ensure it's also copied:
    if isinstance(original_block.get_content(), (list, dict)):
        assert copied_block.get_content() is not original_block.get_content()


def test_message_block_text():
    block = MessageBlock(media=Text("This is a text block"))
    assert block.get_content() == "This is a text block"
    assert len(block) == len("This is a text block")
    assert block.contains("text")
    assert str(block) == "This is a text block"


def test_message_block_object():
    model = ExampleModel(key="example", value="data")
    block = MessageBlock(media=Object(model))
    assert block.get_content() == model
    assert len(block) == len(model.model_dump_json())
    assert block.contains("example")
    assert str(block) == f"ExampleModel: {model}"


def test_message_block_image():
    block = MessageBlock(media=Image("https://example.com/image.jpg"))
    assert block.media.uri == "https://example.com/image.jpg"
    assert len(block) == len("https://example.com/image.jpg")
    assert block.contains("example")
    assert str(block) == "Image: https://example.com/image.jpg"


def test_message_block_audio():
    block = MessageBlock(Audio("https://example.com/audio.mp3"))
    assert block.media.uri == "https://example.com/audio.mp3"
    assert len(block) == len("https://example.com/audio.mp3")
    assert block.contains("example")
    assert str(block) == "Audio: https://example.com/audio.mp3"


def test_message_block_video():
    block = MessageBlock(Video("https://example.com/video.mp4"))
    assert block.media.uri == "https://example.com/video.mp4"
    assert len(block) == len("https://example.com/video.mp4")
    assert block.contains("example")
    assert str(block) == "Video: https://example.com/video.mp4"


def test_message_block_document():
    block = MessageBlock(Document("https://example.com/document.pdf"))
    assert block.media.uri == "https://example.com/document.pdf"
    assert len(block) == len("https://example.com/document.pdf")
    assert block.contains("example")
    assert str(block) == "Document: https://example.com/document.pdf"


def test_message_block_table():
    table = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    block = MessageBlock(Table.from_dataframe(table))
    assert block.has_type(Table)
    assert block.media.to_dataframe().equals(table)
    assert len(block.get_content()) > 40

    assert "Table:" in str(block)
