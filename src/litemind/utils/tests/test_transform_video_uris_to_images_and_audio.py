import pytest

from litemind.agent.messages.message import Message
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_image import Image
from litemind.ressources.media_resources import MediaResources
from litemind.utils.transform_video_uris_to_images_and_audio import (
    transform_video_uris_to_images_and_audio,
)


@pytest.fixture
def create_message_with_video_uri():
    # Combine the two to get the absolute path to 'bunny.mp4'
    video_uri = MediaResources.get_local_test_video_uri("bunny.mp4")

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
    assert any(block.has_type(Image) for block in transformed_message.blocks)
    assert any(block.has_type(Audio) for block in transformed_message.blocks)
