import time
from typing import List, Union

from arbol import aprint
from PIL.Image import Image as PILImageType  # Correct type import

from litemind.agent.messages.actions.action_base import ActionBase
from litemind.agent.messages.actions.tool_call import ToolCall
from litemind.agent.messages.actions.tool_use import ToolUse
from litemind.agent.messages.message import Message
from litemind.media.types.media_action import Action
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video


def convert_messages_for_gemini(
    messages: List[Message],
) -> List[Union[str, PILImageType]]:  # Use PILImageType correctly
    """
    Convert messages into a format suitable for Gemini, supporting both text and image inputs.
    """
    # Upload the video file to Gemini:
    import google.generativeai as genai

    # Initialize the list of messages:
    gemini_messages = []

    # Iterate over each message:
    for message in messages:

        # If the message is a system message, skip it:
        if message.role == "system":
            continue

        # Iterate over each block in the message:
        for block in message.blocks:
            if block.has_type(Text) and not block.is_thinking():
                gemini_messages.append(f"{message.role}: {block.media}")

            elif block.has_type(Text) and block.is_thinking():
                # Thinking blocks are not passed back to the model:
                pass

            elif block.has_type(Image):

                # Cast to Image:
                image: Image = block.media

                try:
                    # Open the image, convert it to PNG format, and append it to the list:
                    with image.open_pil_image(normalise_to_png=True) as img:
                        gemini_messages.append(img)

                except Exception as e:
                    raise ValueError(f"Could not open image '{image.uri}': {e}")

            elif block.has_type(Audio):

                # Cast to Audio:
                audio: Audio = block.media

                try:
                    # Convert the video URI to a local file path:
                    local_path = audio.to_local_file_path()

                    # Upload the audio file to Gemini:
                    genai_audio_file = genai.upload_file(local_path)

                    # Wait for the file to finish processing:
                    while genai_audio_file.state.name == "PROCESSING":
                        time.sleep(0.1)
                        genai_audio_file = genai.get_file(genai_audio_file.name)

                    # Append the uploaded file to the list of messages:
                    gemini_messages.append(genai_audio_file)

                except Exception as e:
                    raise ValueError(f"Could not open audio '{audio.uri}': {e}")

            elif block.has_type(Video):

                # Cast to Video:
                video: Video = block.media

                try:
                    # Convert the video URI to a local file path:
                    local_path = video.to_local_file_path()

                    # Upload the video file to Gemini:
                    genai_video_file = genai.upload_file(local_path)

                    # Wait for the file to finish processing:
                    while genai_video_file.state.name == "PROCESSING":
                        time.sleep(0.1)
                        genai_video_file = genai.get_file(genai_video_file.name)

                    # Append the uploaded file to the list of messages:
                    gemini_messages.append(genai_video_file)

                except Exception as e:
                    raise ValueError(f"Could not process video '{video.uri}': {e}")

            elif block.has_type(Action):

                tool_action: ActionBase = block.get_content()

                # if contents is a ToolUse object do the following:
                if isinstance(tool_action, ToolUse):
                    # Get the tool use object:
                    tool_use: ToolUse = tool_action

                    # Build a functionResponse part
                    func_response = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tool_use.tool_name,
                            response={"result": tool_use.result},
                        )
                    )

                    # Add tool use to the content:
                    gemini_messages.append(func_response)
                elif isinstance(tool_action, ToolCall):
                    # Ignoring tool calls.
                    pass

                else:
                    raise ValueError(
                        f"Unsupported action type: {type(tool_action).__name__}"
                    )

            else:
                raise ValueError(f"Unsupported block type: {block.get_type()}")

    return gemini_messages


def list_and_delete_uploaded_files():
    import google.generativeai as genai

    for f in genai.list_files():
        if f.mime_type.startswith("video/"):
            aprint("  ", f.name)
            f.delete()
            aprint(f"Deleted {f.name}")
