import copy
from pprint import pprint

import pytest
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video

_remote_csv_file = "https://www.sample-videos.com/csv/Sample-Spreadsheet-100-rows.csv"


class StructuredText(BaseModel):
    key: str
    value: str


def test_message_deepcopy():
    # Create a message and add some blocks
    original_message = Message(role="user")
    block1 = MessageBlock(Text("First block"))
    block2 = MessageBlock(Text("Second block"))
    original_message.append_block(block1)
    original_message.append_block(block2)

    # Perform a deep copy of the message
    copied_message = copy.deepcopy(original_message)

    # Verify that the copied message is equal to the original
    assert copied_message.role == original_message.role
    assert len(copied_message) == len(original_message)
    assert copied_message[0].media.text == original_message[0].media.text
    assert copied_message[1].media.text == original_message[1].media.text

    # Verify that the copied message is a different object
    assert copied_message is not original_message
    assert copied_message.blocks is not original_message.blocks
    assert copied_message.blocks[0] is not original_message.blocks[0]
    assert copied_message.blocks[1] is not original_message.blocks[1]


def test_insert_block():
    # Create a few blocks:
    block1 = MessageBlock(Text("First block"))
    block2 = MessageBlock(Text("Second block"))
    block3 = MessageBlock(Text("Inserted block"))

    # Create a user message and append blocks:
    user_message = Message(role="user")
    user_message.append_block(block1)
    user_message.append_block(block2)

    # Insert block at index 1
    user_message.insert_block(block3, block_before=block1)

    assert user_message[0].media.get_content() == "First block"
    assert user_message[1].media.get_content() == "Inserted block"
    assert user_message[2].media.get_content() == "Second block"


def test_insert_message():
    # Create a few messages:
    user_message1 = Message(role="user")
    user_message2 = Message(role="user")

    # Create a few blocks:
    block1 = MessageBlock(Text("First block"))
    block2 = MessageBlock(Text("Second block"))
    block3 = MessageBlock(Text("Third block"))

    # Append blocks to message 1:
    user_message1.append_block(block1)
    user_message1.append_block(block2)

    # Append blocks to message 2:
    user_message2.append_block(block3)

    # Insert message at index 1
    user_message1.insert_message(user_message2, block_before=block1)

    assert user_message1[0].media.get_content() == "First block"
    assert user_message1[1].media.get_content() == "Third block"
    assert user_message1[2].media.get_content() == "Second block"


def test_message_text():
    system_message = Message(role="system")
    system_message.append_text("You are an omniscient all-knowing being called Ohmm")

    assert system_message.role == "system"
    assert any(
        block.media.get_content()
        == "You are an omniscient all-knowing being called Ohmm"
        for block in system_message.blocks
    )

    user_message = Message(role="user")
    user_message.append_text("Who are you?")

    assert user_message.role == "user"
    assert any(block.get_content() == "Who are you?" for block in user_message.blocks)

    # Checks contains operator:
    assert "Who" in user_message
    assert "omniscient" in system_message


def test_append_templated_text_success():
    user_message = Message(role="user")
    block = user_message.append_templated_text("Hello, {name}!", name="Alice")
    assert "Hello, Alice!" in str(block.media.text)
    assert isinstance(block, type(user_message.blocks[0]))


def test_append_templated_text_missing_replacement():
    user_message = Message(role="user")
    with pytest.raises(KeyError):
        user_message.append_templated_text("Hello, {name} and {other}!", name="Alice")


def test_message_object():
    obj = StructuredText(key="example", value="data")
    user_message = Message(role="user")
    user_message.append_object(obj)

    assert user_message.role == "user"
    assert any(block.get_content() == obj for block in user_message.blocks)


def test_message_json():
    obj = StructuredText(key="example", value="data")

    # Convert pydantic obj to json:
    json = obj.model_dump_json()

    user_message = Message(role="user")
    user_message.append_json(json)

    assert user_message.role == "user"
    assert user_message.blocks[0].get_content()["key"] == "example"
    assert user_message.blocks[0].get_content()["value"] == "data"


def test_message_image():
    user_message = Message(role="user")
    user_message.append_text("Can you describe what you see?")
    user_message.append_image("https://example.com/image.jpg")

    assert user_message.role == "user"
    assert any(
        block.get_content() == "Can you describe what you see?"
        for block in user_message.blocks
    )
    assert any(
        block.get_content() == "https://example.com/image.jpg" and block.has_type(Image)
        for block in user_message.blocks
    )

    # Checks contains operator:
    assert "example" in user_message


def test_message_audio():
    user_message = Message(role="user")
    user_message.append_text("Can you describe what you hear?")
    user_message.append_audio("https://example.com/audio.mp3")

    assert user_message.role == "user"
    assert any(
        block.get_content() == "Can you describe what you hear?"
        for block in user_message.blocks
    )
    assert any(
        block.get_content() == "https://example.com/audio.mp3" and block.has_type(Audio)
        for block in user_message.blocks
    )

    # Checks contains operator:
    assert "example" in user_message


def test_message_video():
    user_message = Message(role="user")
    user_message.append_text("Can you describe what you see in the video?")
    user_message.append_video("https://example.com/video.mp4")

    assert user_message.role == "user"
    assert any(
        block.get_content() == "Can you describe what you see in the video?"
        for block in user_message.blocks
    )
    assert any(
        block.get_content() == "https://example.com/video.mp4" and block.has_type(Video)
        for block in user_message.blocks
    )

    # Checks contains operator:
    assert "example" in user_message


def test_message_document():
    user_message = Message(role="user")
    user_message.append_text("Can you describe what you see in the document?")
    user_message.append_document("https://example.com/document.pdf")

    assert user_message.role == "user"
    assert any(
        block.get_content() == "Can you describe what you see in the document?"
        for block in user_message.blocks
    )
    assert any(
        block.get_content() == "https://example.com/document.pdf"
        and block.has_type(Document)
        for block in user_message.blocks
        for block in user_message.blocks
    )

    # Checks contains operator:
    assert "example" in user_message


def test_message_table():
    from pandas import DataFrame

    # Create a small table with pandas:
    table = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    user_message = Message(role="user")
    user_message.append_text("Can you describe what you see in the table?")
    user_message.append_table(table)

    assert user_message.role == "user"
    assert any(
        block.get_content() == "Can you describe what you see in the table?"
        for block in user_message.blocks
    )

    # Check that we can get a DataFrame:
    assert isinstance(user_message[1].media.to_dataframe(), DataFrame)

    # Checks contains operator:
    assert "table" in user_message

    # Now tests ith the remote UL of a CSV file:

    user_message = Message(role="user")
    user_message.append_text("Can you describe what you see in the table?")
    user_message.append_table(_remote_csv_file)

    assert user_message.role == "user"
    assert any(
        block.get_content() == "Can you describe what you see in the table?"
        for block in user_message.blocks
    )
    assert isinstance(user_message[1].media.to_dataframe(), DataFrame)


def test_message_folder():
    # Create an empty temp folder with tempfile:
    import os
    import tempfile

    temp_folder = tempfile.mkdtemp()

    # Create a text file with random sentence in the temp folder:
    with open(os.path.join(temp_folder, "file.txt"), "w") as f:
        f.write("This is a random sentence.")

    # Create an image file (PNG) with some random pixel values use Pillow:
    import numpy as np
    from PIL import Image as Image_PIL

    image = Image_PIL.fromarray(
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    )
    image.save(os.path.join(temp_folder, "image.png"))

    # Download this PDF into the folder: https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf
    import requests

    pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    pdf_file = os.path.join(temp_folder, "document.pdf")
    with open(pdf_file, "wb") as f:
        f.write(requests.get(pdf_url).content)

    # Add other different types of files to the folder:
    os.makedirs(os.path.join(temp_folder, "folder1", "folder2"))
    with open(os.path.join(temp_folder, "folder1", "file1.txt"), "w") as f:
        f.write("This is a random sentence 1.")
    with open(os.path.join(temp_folder, "folder1", "file2.txt"), "w") as f:
        f.write("This is a random sentence 2.")
    with open(os.path.join(temp_folder, "folder1", "folder2", "file3.txt"), "w") as f:
        f.write("This is a random sentence 3.")

    # Add a hidden file (starts with '.') to the folder:
    with open(os.path.join(temp_folder, ".hidden_file.txt"), "w") as f:
        f.write("This is a hidden file.")

    # Adds a hidden folder (starts with '.') to the folder:
    os.makedirs(os.path.join(temp_folder, ".hidden_folder"))

    # Add an empty file to the folder:
    with open(os.path.join(temp_folder, "empty_file.txt"), "w") as f:
        pass

    user_message = Message(role="user")
    user_message.append_text("Can you describe what you see in the folder?")
    user_message.append_folder(temp_folder)

    pprint(user_message)

    assert user_message.role == "user"

    # Check that the message has a folder tree:
    assert any(
        block.get_content() == "Can you describe what you see in the folder?"
        for block in user_message.blocks
    )

    # Check that the message has a folder tree and specific files:
    assert any(
        block.get_content() == "Can you describe what you see in the folder?"
        for block in user_message.blocks
    )
    assert "file.txt" in str(user_message)
    assert "image.png" in str(user_message)
    assert "document.pdf" in str(user_message)

    # Check the presence of a document block and an image block:
    assert any(block.has_type(Document) for block in user_message.blocks)
    assert any(block.has_type(Image) for block in user_message.blocks)

    # Check that the files are correctly represented in the message:
    assert "Image File: image.png" in str(user_message)
    assert "Document File: document.pdf" in str(user_message)
    assert "Empty File: empty_file.txt" in str(user_message)
    assert "Text File: file.txt" in str(user_message)
    assert "Text File: file2.txt" in str(user_message)
    assert "Text File: file3.txt" in str(user_message)


def test_message_contains():
    user_message = Message(role="user")
    user_message.append_text("Can you describe what you see?")
    user_message.append_image("https://example.com/image.jpg")
    user_message.append_audio("https://example.com/audio.mp3")
    user_message.append_video("https://example.com/video.mp4")
    user_message.append_document("https://example.com/document.pdf")
    user_message.append_table(_remote_csv_file)

    assert "example" in user_message
    assert "data" not in user_message
    assert "image" in user_message
    assert "audio" in user_message
    assert "video" in user_message
    assert "document" in user_message
    assert "Table" in user_message


def test_message_str():
    user_message = Message(role="user")
    user_message.append_text("Can you describe what you see?")
    user_message.append_image("https://example.com/image.jpg")
    user_message.append_audio("https://example.com/audio.mp3")
    user_message.append_video("https://example.com/video.mp4")
    user_message.append_document("https://example.com/document.pdf")
    user_message.append_table(_remote_csv_file)

    assert "Can you describe what you see?" in str(user_message)
    assert "https://example.com/image.jpg" in str(user_message)
    assert "https://example.com/audio.mp3" in str(user_message)
    assert "https://example.com/video.mp4" in str(user_message)
    assert "https://example.com/document.pdf" in str(user_message)
    assert _remote_csv_file in str(user_message)


def test_extract_markdown_block():
    # Create a message and add some blocks
    user_message = Message(role="user")
    user_message.append_text(
        "This is a text block:\n" "```markdown\n# Header\nSome content\n```"
    )
    user_message.append_text("Another text block")
    user_message.append_text("```markdown\n# Another Header\nMore content\n```")

    # Extract markdown blocks containing the filter string 'Header'
    filters = ["Header"]
    markdown_blocks = user_message.extract_markdown_block(filters, remove_quotes=False)

    # Verify that the extracted blocks are correct
    assert len(markdown_blocks) == 2
    assert (
        markdown_blocks[0].get_content() == "```markdown\n# Header\nSome content\n```"
    )
    assert (
        markdown_blocks[1].get_content()
        == "```markdown\n# Another Header\nMore content\n```"
    )

    # We repeat but we remove the quotes:
    markdown_blocks = user_message.extract_markdown_block(filters, remove_quotes=True)

    # Verify that the extracted blocks are correct
    assert len(markdown_blocks) == 2
    assert markdown_blocks[0].get_content() == "# Header\nSome content\n"
    assert markdown_blocks[1].get_content() == "# Another Header\nMore content\n"


def test_list_present_media_types():
    # Create a message and add some blocks
    user_message = Message(role="user")
    user_message.append_text("This is a text block")
    user_message.append_image("https://example.com/image.jpg")
    user_message.append_audio("https://example.com/audio.mp3")
    user_message.append_video("https://example.com/video.mp4")
    user_message.append_document("https://example.com/document.pdf")

    # List present media types
    media_types = user_message.list_media_types()

    # Verify that the media types are correct
    assert Text in media_types
    assert Video in media_types
    assert Audio in media_types
    assert Image in media_types
    assert Document in media_types
