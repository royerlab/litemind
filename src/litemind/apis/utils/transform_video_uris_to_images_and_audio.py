from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.apis.utils.ffmpeg_utils import convert_video_to_frames_and_audio
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


def transform_video_uris_to_images_and_audio(message: Message) -> Message:
    """
    Convert video URIs in the message to images and audio blocks.

    Parameters
    ----------
    message : Message
        The message object containing video URIs.

    Returns
    -------
    Message
        The message object with video URIs converted to images and audio blocks.
    """
    # Iterate over each block in the message:
    for block in message.blocks:
        if block.block_type == BlockType.Video and block.content is not None:
            # Normalise the URI to a local file path:
            local_path = uri_to_local_file_path(block.content)

            # Get video frames and audio:
            blocks = convert_video_to_frames_and_audio(local_path)

            # Insert the new blocks at the current position:
            message.insert_blocks(blocks, block_before=block)

            # Remove the original video block:
            message.blocks.remove(block)

    return message
