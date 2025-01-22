from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType
from litemind.apis.utils.sample_video import \
    append_video_frames_and_audio_to_message
from litemind.apis.utils.write_base64_to_temp_file import \
    write_base64_to_temp_file
from litemind.utils.normalise_uri_to_local_file_path import \
    uri_to_local_file_path


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
        if block.block_type == BlockType.Video and block.content:
            if block.content.startswith("data:video/"):
                local_path = write_base64_to_temp_file(block.content)
            elif block.content.startswith(
                    "http://") or block.content.startswith("https://"):
                local_path = uri_to_local_file_path(block.content)
            elif block.content.startswith("file://"):
                local_path = block.content.replace("file://", "")
            else:
                raise ValueError(
                    f"Invalid video URI: '{block.content}' (must start with 'data:video/', 'http://', 'https://', or 'file://')"
                )

            # Append video frames and audio to the message:
            message = append_video_frames_and_audio_to_message(local_path,
                                                               message)

    return message
