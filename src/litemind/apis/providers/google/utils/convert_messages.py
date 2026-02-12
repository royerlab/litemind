"""Conversion of litemind Messages to Gemini Content objects."""

import io
import time
from typing import TYPE_CHECKING, List

from litemind.agent.messages.actions.action_base import ActionBase
from litemind.agent.messages.actions.tool_call import ToolCall
from litemind.agent.messages.actions.tool_use import ToolUse
from litemind.agent.messages.message import Message
from litemind.media.types.media_action import Action
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video

if TYPE_CHECKING:
    from google.genai import Client


def convert_messages_for_gemini(
    messages: List[Message],
    client: "Client",
) -> list:
    """Convert litemind Messages into Gemini Content objects.

    Produces a list of ``types.Content`` objects with proper ``role``
    assignments (``user`` / ``model``) so that thinking-model thought
    signatures are preserved across turns.

    System messages are skipped since Gemini handles them separately
    via the ``system_instruction`` config parameter.

    Parameters
    ----------
    messages : List[Message]
        Litemind messages to convert.
    client : Client
        The google-genai Client instance, used for uploading audio/video files.

    Returns
    -------
    list
        List of ``types.Content`` objects suitable for Gemini's
        ``generate_content`` method.
    """
    from google.genai import types

    gemini_contents: list = []

    for message in messages:
        # System messages are handled via system_instruction in the config:
        if message.role == "system":
            continue

        # Map litemind roles to Gemini roles:
        gemini_role = "model" if message.role == "assistant" else "user"

        # If this is a model message with preserved raw parts (including
        # thought signatures), replay them directly to maintain context
        # for Gemini thinking models across turns:
        raw_parts = getattr(message, "_gemini_raw_parts", None)
        if raw_parts and gemini_role == "model":
            gemini_contents.append(types.Content(role="model", parts=raw_parts))
            continue

        # Collect parts for this message:
        parts: list = []

        for block in message.blocks:
            if block.has_type(Text) and not block.is_thinking():
                parts.append(types.Part(text=str(block.media)))

            elif block.has_type(Text) and block.is_thinking():
                # Thinking blocks are not passed back to the model:
                pass

            elif block.has_type(Image):
                image: Image = block.media
                try:
                    with image.open_pil_image(normalise_to_png=True) as img:
                        img_copy = img.copy()
                        buf = io.BytesIO()
                        img_copy.save(buf, format="PNG")
                        parts.append(
                            types.Part.from_bytes(
                                data=buf.getvalue(),
                                mime_type="image/png",
                            )
                        )
                except Exception as e:
                    raise ValueError(f"Could not open image '{image.uri}': {e}")

            elif block.has_type(Audio):
                audio: Audio = block.media
                try:
                    local_path = audio.to_local_file_path()
                    genai_audio_file = client.files.upload(file=local_path)
                    while genai_audio_file.state.name == "PROCESSING":
                        time.sleep(0.1)
                        genai_audio_file = client.files.get(name=genai_audio_file.name)
                    parts.append(
                        types.Part.from_uri(
                            file_uri=genai_audio_file.uri,
                            mime_type=genai_audio_file.mime_type,
                        )
                    )
                except Exception as e:
                    raise ValueError(f"Could not open audio '{audio.uri}': {e}")

            elif block.has_type(Video):
                video: Video = block.media
                try:
                    local_path = video.to_local_file_path()
                    genai_video_file = client.files.upload(file=local_path)
                    while genai_video_file.state.name == "PROCESSING":
                        time.sleep(0.1)
                        genai_video_file = client.files.get(name=genai_video_file.name)
                    parts.append(
                        types.Part.from_uri(
                            file_uri=genai_video_file.uri,
                            mime_type=genai_video_file.mime_type,
                        )
                    )
                except Exception as e:
                    raise ValueError(f"Could not process video '{video.uri}': {e}")

            elif block.has_type(Action):
                tool_action: ActionBase = block.get_content()

                if isinstance(tool_action, ToolUse):
                    tool_use: ToolUse = tool_action
                    func_response = types.Part.from_function_response(
                        name=tool_use.tool_name,
                        response={"result": tool_use.result},
                    )
                    parts.append(func_response)
                elif isinstance(tool_action, ToolCall):
                    tool_call: ToolCall = tool_action
                    func_call = types.Part.from_function_call(
                        name=tool_call.tool_name,
                        args=tool_call.arguments,
                    )
                    parts.append(func_call)
                else:
                    raise ValueError(
                        f"Unsupported action type: {type(tool_action).__name__}"
                    )

            else:
                raise ValueError(f"Unsupported block type: {block.get_type()}")

        # Only add content if there are parts:
        if parts:
            gemini_contents.append(types.Content(role=gemini_role, parts=parts))

    return gemini_contents
