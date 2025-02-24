import os

import pytest

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.apis.utils.transform_video_uris_to_images_and_audio import (
    transform_video_uris_to_images_and_audio,
)


@pytest.fixture
def create_message_with_video_uri():
    # Combine the two to get the absolute path to 'harvard.wav'
    current_dir = os.path.dirname(__file__)

    # Combine the two to get the absolute path to 'flying.mp4'
    video_path = os.path.join(current_dir, "media/bunny.mp4")

    # Convert to URI:
    video_uri = f"file://{video_path}"

    # Create a message with only a video URI:
    message = Message(role="user")
    message.append_video(video_uri)

    return message


def test_transform_video_uris_to_images_and_audio(create_message_with_video_uri):
    message = create_message_with_video_uri

    # Transform the message:
    transformed_message = transform_video_uris_to_images_and_audio(message)

    # Check if the message has been transformed correctly
    assert transformed_message is not None
    assert len(transformed_message.blocks) > 1
    assert any(
        block.block_type == BlockType.Image for block in transformed_message.blocks
    )
    assert any(
        block.block_type == BlockType.Audio for block in transformed_message.blocks
    )
