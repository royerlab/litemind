"""Utilities for converting video message blocks into image frames and audio."""

from litemind.agent.messages.message import Message
from litemind.media.types.media_video import Video


def transform_video_uris_to_images_and_audio(message: Message) -> Message:
    """
    Convert video URIs in a message to image frames and audio blocks.

    Iterates over each block in the message, and for any block containing
    a video, extracts image frames and audio, replacing the original
    video block with the extracted content.

    Parameters
    ----------
    message : Message
        The message object containing video blocks to convert.

    Returns
    -------
    Message
        The same message object with video blocks replaced by image
        and audio blocks.
    """
    # Iterate over each block in the message:
    for block in message.blocks:
        if block.has_type(Video) and block.get_content() is not None:
            # cast to Video:
            video: Video = block.media

            # Get video frames and audio:
            blocks = video.convert_to_frames_and_audio()

            # Insert the new blocks at the current position:
            message.insert_blocks(blocks, block_before=block)

            # Remove the original video block:
            message.blocks.remove(block)

    return message
