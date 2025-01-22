import pathlib
import time
from typing import List, Union

from PIL import Image
from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType
from litemind.utils.normalise_uri_to_local_file_path import \
    uri_to_local_file_path


def _convert_messages_for_gemini(messages: List[Message]) -> List[
    Union[str, Image.Image]]:
    """
    Convert messages into a format suitable for Gemini, supporting both text and image inputs.
    """

    # Initialize the list of messages:
    gemini_messages = []

    # Iterate over each message:
    for message in messages:

        # Iterate over each block in the message:
        for block in message.blocks:
            if block.block_type == BlockType.Text:
                gemini_messages.append(f"{message.role}: {block.content}")
            elif block.block_type == BlockType.Image:
                image_uri = block.content
                try:
                    # Convert the image URI to a local file path:
                    local_path = uri_to_local_file_path(image_uri)

                    # Open the image:
                    with Image.open(local_path) as img:
                        gemini_messages.append(img)
                except Exception as e:
                    raise ValueError(f"Could not open image '{image_uri}': {e}")
            elif block.block_type == BlockType.Audio:
                audio_uri = block.content
                try:
                    # Convert the audio URI to a local file path:
                    local_path = uri_to_local_file_path(audio_uri)

                    # Read the audio data:
                    audio_data = pathlib.Path(local_path).read_bytes()

                    # Determine the MIME type:
                    mime_type = "audio/" + local_path.split('.')[-1]

                    # Append the audio data to the list of messages:
                    gemini_messages.append(
                        {"mime_type": mime_type, "data": audio_data})
                except Exception as e:
                    raise ValueError(f"Could not open audio '{audio_uri}': {e}")
            elif block.block_type == BlockType.Video:
                video_uri = block.content
                try:
                    # Convert the video URI to a local file path:
                    local_path = uri_to_local_file_path(video_uri)

                    # Upload the video file to Gemini:
                    import google.generativeai as genai
                    myfile = genai.upload_file(local_path)

                    # Wait for the file to finish processing:
                    while myfile.state.name == "PROCESSING":
                        time.sleep(0.1)
                        myfile = genai.get_file(myfile.name)

                    # Append the uploaded file to the list of messages:
                    gemini_messages.append(myfile)
                except Exception as e:
                    raise ValueError(
                        f"Could not process video '{video_uri}': {e}")

    return gemini_messages


def _list_and_delete_uploaded_files():
    import google.generativeai as genai
    for f in genai.list_files():
        if f.mime_type.startswith("video/"):
            aprint("  ", f.name)
            f.delete()
            aprint(f"Deleted {f.name}")
