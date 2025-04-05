from litemind.agent.messages.message import Message
from litemind.media.types.media_video import Video


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
