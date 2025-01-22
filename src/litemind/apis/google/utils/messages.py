import pathlib
import time
from typing import List, Union

from PIL import Image
from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType
from litemind.apis.utils.write_base64_to_temp_file import \
    write_base64_to_temp_file
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
                    if image_uri.startswith("data:image/"):
                        local_path = write_base64_to_temp_file(image_uri)
                    elif image_uri.startswith(
                            "http://") or image_uri.startswith("https://"):
                        local_path = uri_to_local_file_path(image_uri)
                    elif image_uri.startswith("file://"):
                        local_path = image_uri.replace("file://", "")
                    else:
                        raise ValueError(f"Invalid image URI: '{image_uri}'")

                    with Image.open(local_path) as img:
                        gemini_messages.append(img)
                except Exception as e:
                    raise ValueError(f"Could not open image '{image_uri}': {e}")
            elif block.block_type == BlockType.Audio:
                audio_uri = block.content
                try:
                    if audio_uri.startswith("data:audio/"):
                        local_path = write_base64_to_temp_file(audio_uri)
                    elif audio_uri.startswith(
                            "http://") or audio_uri.startswith("https://"):
                        local_path = uri_to_local_file_path(audio_uri)
                    elif audio_uri.startswith("file://"):
                        local_path = audio_uri.replace("file://", "")
                    else:
                        raise ValueError(f"Invalid audio URI: '{audio_uri}'")

                    audio_data = pathlib.Path(local_path).read_bytes()
                    mime_type = "audio/" + local_path.split('.')[-1]
                    gemini_messages.append(
                        {"mime_type": mime_type, "data": audio_data})
                except Exception as e:
                    raise ValueError(f"Could not open audio '{audio_uri}': {e}")
            elif block.block_type == BlockType.Video:
                video_uri = block.content
                try:
                    if video_uri.startswith("data:video/"):
                        local_path = write_base64_to_temp_file(video_uri)
                    elif video_uri.startswith(
                            "http://") or video_uri.startswith("https://"):
                        local_path = uri_to_local_file_path(video_uri)
                    elif video_uri.startswith("file://"):
                        local_path = video_uri.replace("file://", "")
                    else:
                        raise ValueError(f"Invalid video URI: '{video_uri}'")

                    import google.generativeai as genai
                    myfile = genai.upload_file(local_path)
                    while myfile.state.name == "PROCESSING":
                        time.sleep(0.1)
                        myfile = genai.get_file(myfile.name)

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
