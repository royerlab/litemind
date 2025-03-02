import pathlib
import time
from typing import List, Union

from arbol import aprint
from PIL import Image

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.messages.tool_use import ToolUse
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


def convert_messages_for_gemini(
    messages: List[Message],
) -> List[Union[str, Image.Image]]:
    """
    Convert messages into a format suitable for Gemini, supporting both text and image inputs.
    """
    # Upload the video file to Gemini:
    import google.generativeai as genai

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
                    mime_type = "audio/" + local_path.split(".")[-1]

                    # Append the audio data to the list of messages:
                    gemini_messages.append({"mime_type": mime_type, "data": audio_data})
                except Exception as e:
                    raise ValueError(f"Could not open audio '{audio_uri}': {e}")
            elif block.block_type == BlockType.Video:
                video_uri = block.content
                try:
                    # Convert the video URI to a local file path:
                    local_path = uri_to_local_file_path(video_uri)

                    genai_video_file = genai.upload_file(local_path)

                    # Wait for the file to finish processing:
                    while genai_video_file.state.name == "PROCESSING":
                        time.sleep(0.1)
                        genai_video_file = genai.get_file(genai_video_file.name)

                    # Append the uploaded file to the list of messages:
                    gemini_messages.append(genai_video_file)
                except Exception as e:
                    raise ValueError(f"Could not process video '{video_uri}': {e}")

            elif block.block_type == BlockType.Tool:

                # if contents is a ToolUse object do the following:
                if isinstance(block.content, ToolUse):

                    # Get the tool use object:
                    tool_use: ToolUse = block.content

                    # Build a functionResponse part
                    func_response = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_use.tool_name,
                            response={"result": tool_use.result},
                        )
                    )

                    # Add tool use to the content:
                    gemini_messages.append(func_response)

            else:
                raise ValueError(f"Unsupported block type: {block.block_type}")

    return gemini_messages


def list_and_delete_uploaded_files():
    import google.generativeai as genai

    for f in genai.list_files():
        if f.mime_type.startswith("video/"):
            aprint("  ", f.name)
            f.delete()
            aprint(f"Deleted {f.name}")
