from pydantic import BaseModel

from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType

_remote_csv_file = 'https://www.sample-videos.com/csv/Sample-Spreadsheet-100-rows.csv'


class StructuredText(BaseModel):
    key: str
    value: str


def test_message_text():
    system_message = Message(role='system')
    system_message.append_text(
        'You are an omniscient all-knowing being called Ohmm')

    assert system_message.role == 'system'
    assert any(
        block.content == 'You are an omniscient all-knowing being called Ohmm'
        for block in system_message.blocks)

    user_message = Message(role='user')
    user_message.append_text('Who are you?')

    assert user_message.role == 'user'
    assert any(block.content == 'Who are you?' for block in user_message.blocks)

    # Checks contains operator:
    assert 'Who' in user_message
    assert 'omniscient' in system_message


def test_message_object():
    obj = StructuredText(key='example', value='data')
    user_message = Message(role='user')
    user_message.append_object(obj)

    assert user_message.role == 'user'
    assert any(block.content == obj for block in user_message.blocks)


def test_message_json():
    obj = StructuredText(key='example', value='data')

    # Convert pydantic obj to json:
    json = obj.model_dump_json()

    user_message = Message(role='user')
    user_message.append_json(json)

    assert user_message.role == 'user'
    assert any(block.content == json for block in user_message.blocks)


def test_message_image():
    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see?')
    user_message.append_image('https://example.com/image.jpg')

    assert user_message.role == 'user'
    assert any(block.content == 'Can you describe what you see?' for block in
               user_message.blocks)
    assert any(
        block.content == 'https://example.com/image.jpg' and block.block_type == BlockType.Image
        for block in user_message.blocks)

    # Checks contains operator:
    assert 'example' in user_message


def test_message_audio():
    user_message = Message(role='user')
    user_message.append_text('Can you describe what you hear?')
    user_message.append_audio('https://example.com/audio.mp3')

    assert user_message.role == 'user'
    assert any(block.content == 'Can you describe what you hear?' for block in
               user_message.blocks)
    assert any(
        block.content == 'https://example.com/audio.mp3' and block.block_type == BlockType.Audio
        for block in user_message.blocks)

    # Checks contains operator:
    assert 'example' in user_message


def test_message_video():
    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see in the video?')
    user_message.append_video('https://example.com/video.mp4')

    assert user_message.role == 'user'
    assert any(
        block.content == 'Can you describe what you see in the video?' for block
        in user_message.blocks)
    assert any(
        block.content == 'https://example.com/video.mp4' and block.block_type == BlockType.Video
        for block in user_message.blocks)

    # Checks contains operator:
    assert 'example' in user_message


def test_message_document():
    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see in the document?')
    user_message.append_document('https://example.com/document.pdf')

    assert user_message.role == 'user'
    assert any(
        block.content == 'Can you describe what you see in the document?' for
        block in user_message.blocks)
    assert any(
        block.content == 'https://example.com/document.pdf' and block.block_type == BlockType.Document
        for block in user_message.blocks)

    # Checks contains operator:
    assert 'example' in user_message


def test_message_table():
    from pandas import DataFrame

    # Create a small table with pandas:
    table = DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see in the table?')
    user_message.append_table(table)

    assert user_message.role == 'user'
    assert any(
        block.content == 'Can you describe what you see in the table?' for block
        in user_message.blocks)

    # Check that the table is in the message with its specific values:
    assert isinstance(user_message[1].content, DataFrame)

    # Checks contains operator:
    assert 'table' in user_message

    # Now tests ith the remote UL of a CSV file:

    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see in the table?')
    user_message.append_table(_remote_csv_file)

    assert user_message.role == 'user'
    assert any(
        block.content == 'Can you describe what you see in the table?' for block
        in user_message.blocks)
    assert isinstance(user_message[1].content, DataFrame)


def test_message_folder():
    # Create an empty temp folder with tempfile:
    import tempfile
    import os
    temp_folder = tempfile.mkdtemp()

    # Create a text file with random sentence in the temp folder:
    with open(os.path.join(temp_folder, 'file.txt'), 'w') as f:
        f.write('This is a random sentence.')

    # Create an image file (PNG) with some random pixel values use Pillow:
    import numpy as np
    from PIL import Image
    image = Image.fromarray(
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    image.save(os.path.join(temp_folder, 'image.png'))

    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see in the folder?')
    user_message.append_folder(temp_folder)


def test_message_contains():
    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see?')
    user_message.append_image('https://example.com/image.jpg')
    user_message.append_audio('https://example.com/audio.mp3')
    user_message.append_video('https://example.com/video.mp4')
    user_message.append_document('https://example.com/document.pdf')
    user_message.append_table(_remote_csv_file)

    assert 'example' in user_message
    assert 'data' not in user_message
    assert 'image' in user_message
    assert 'audio' in user_message
    assert 'video' in user_message
    assert 'document' in user_message
    assert '0.58' in user_message


def test_message_str():
    user_message = Message(role='user')
    user_message.append_text('Can you describe what you see?')
    user_message.append_image('https://example.com/image.jpg')
    user_message.append_audio('https://example.com/audio.mp3')
    user_message.append_video('https://example.com/video.mp4')
    user_message.append_document('https://example.com/document.pdf')
    user_message.append_table(_remote_csv_file)

    assert 'Can you describe what you see?' in str(user_message)
    assert 'https://example.com/image.jpg' in str(user_message)
    assert 'https://example.com/audio.mp3' in str(user_message)
    assert 'https://example.com/video.mp4' in str(user_message)
    assert 'https://example.com/document.pdf' in str(user_message)
    assert '0.58' in str(user_message)
