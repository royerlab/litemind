import pathlib
from typing import List, Union

from PIL import Image

from litemind.agent.message import Message
from litemind.apis.utils.dowload_audio_to_tempfile import \
    download_audio_to_temp_file
from litemind.apis.utils.dowload_image_to_tempfile import \
    download_image_to_temp_file
from litemind.apis.utils.write_base64_to_temp_file import \
    write_base64_to_temp_file


def _convert_messages_for_gemini(messages: List[Message]) -> List[
    Union[str, Image.Image]]:
    """
    Convert messages into a format suitable for Gemini, supporting both text and image inputs.
    """

    # Initialize the list of messages:
    gemini_messages = []

    # Iterate over each message:
    for msg in messages:

        # Append the text content:
        if msg.text:
            gemini_messages.append(f"{msg.role}: {msg.text}")

        # Append the image content:
        for image_uri in msg.image_uris:

            # Google API requires the image to be PIL format,
            # so the first step is to get the image on a local file:

            try:

                if image_uri.startswith("data:image/"):
                    # If it's a data URI, put the base64 data in base64_data:
                    local_path = write_base64_to_temp_file(image_uri)

                elif image_uri.startswith("http://") or image_uri.startswith(
                        "https://"):
                    # If it's a remote URL, download the image to a temp file:
                    local_path = download_image_to_temp_file(image_uri)

                elif image_uri.startswith("file://"):
                    # If it's a local file path, then we are good
                    local_path = image_uri.replace("file://", "")

                else:
                    # Raise an error if the image URI is invalid:
                    raise ValueError(f"Invalid image URI: '{image_uri}'")

                # Open the image and append it to the list of messages:
                with Image.open(local_path) as img:
                    gemini_messages.append(img)

            except Exception as e:
                raise ValueError(f"Could not open image '{image_uri}': {e}")

        # Append the audio content:
        for audio_uri in msg.audio_uris:
            try:
                if audio_uri.startswith("data:audio/"):
                    local_path = write_base64_to_temp_file(audio_uri)
                elif audio_uri.startswith(
                        "http://") or audio_uri.startswith("https://"):
                    local_path = download_audio_to_temp_file(audio_uri)
                elif audio_uri.startswith("file://"):
                    local_path = audio_uri.replace("file://", "")
                else:
                    raise ValueError(f"Invalid audio URI: '{audio_uri}'")

                # Read the audio file and append it as a dictionary with mime_type and data
                audio_data = pathlib.Path(local_path).read_bytes()
                mime_type = "audio/" + local_path.split('.')[-1]
                gemini_messages.append(
                    {"mime_type": mime_type, "data": audio_data})
            except Exception as e:
                raise ValueError(f"Could not open audio '{audio_uri}': {e}")

    return gemini_messages
