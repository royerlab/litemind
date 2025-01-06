from typing import List

from openai import OpenAI

from litemind.agent.message import Message
from litemind.apis.utils.dowload_audio_to_tempfile import \
    download_audio_to_temp_file
from litemind.apis.utils.write_base64_to_temp_file import \
    write_base64_to_temp_file


def transcribe_audio_in_messages(messages: List[Message],
                                 client: OpenAI) -> List[Message]:
    # Iterate over each message in the list:
    for message in messages:

        # If the message has audio_uris:
        if message.audio_uris:

            # Iterate over each audio URI in the message:
            for audio_uri in message.audio_uris:
                try:
                    # Save original filename fromm URI:
                    original_filename = audio_uri.split("/")[-1]

                    # Remove the "file://" prefix if it exists:
                    if audio_uri.startswith("file://"):
                        audio_uri = audio_uri.replace("file://", "")

                    # if the audio_uri is a remote url:
                    if audio_uri.startswith("http://") or audio_uri.startswith(
                            "https://"):
                        # Download the audio file:
                        audio_uri = download_audio_to_temp_file(audio_uri)

                    # if the audio_uri is a data uri:
                    elif audio_uri.startswith("data:audio/"):
                        # Write the audio data to a temp file:
                        audio_uri = write_base64_to_temp_file(audio_uri)

                    # Open the audio file:
                    with open(audio_uri, "rb") as audio_file:

                        # Transcribe the audio file:
                        transcription = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="text"
                        )

                        # Add markdown quotes ''' around the transcribed text, and
                        # add prefix: "Transcription: " to the transcribed text:
                        transcription = f"\nTranscription of audio file '{original_filename}': \n'''\n{transcription}\n'''\n"

                        # Add the transcribed text to the message
                        message.append_text(transcription)
                except Exception as e:
                    raise ValueError(
                        f"Could not transcribe audio '{audio_uri}': {e}")

            # If the audio was transcribed, remove the audio_uris from the message:
            message.audio_uris = []

    return messages
