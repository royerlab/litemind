from litemind.utils.dowload_audio_to_tempfile import \
    download_audio_to_temp_file


def is_local_whisper_available() -> bool:
    """
    Check if Whisper is available.

    :return: True if Whisper is available, False otherwise.
    """

    try:
        import whisper
        return True
    except ImportError:
        return False


def transcribe_audio_with_local_whisper(audio_uri: str,
                                        model_name: str = "turbo") -> str:
    """
    Transcribe audio using a local instance of Whisper.

    :param audio_uri: URI of the audio file.
    :param model_name: Name of the Whisper model to use.
    :return: Transcribed text.
    """

    # Import Whisper here to avoid circular imports
    import whisper

    # Download the audio file if it's a remote URL
    if audio_uri.startswith("http://") or audio_uri.startswith("https://"):
        local_path = download_audio_to_temp_file(audio_uri)
    elif audio_uri.startswith("file://"):
        local_path = audio_uri.replace("file://", "")
    else:
        local_path = audio_uri

    # Load the Whisper model
    model = whisper.load_model(model_name)

    # Transcribe the audio file
    result = model.transcribe(local_path)

    return result["text"]

#
# def transcribe_audio_in_messages_with_whisper(messages: List[Message],
#                                               client: Optional[Any] = None) -> \
# List[Message]:
#     from openai import OpenAI
#     client: OpenAI = client
#
#     # Iterate over each message in the list:
#     for message in messages:
#
#         # If the message has audio_uris:
#         if message.audio_uris:
#
#             # Iterate over each audio URI in the message:
#             for audio_uri in message.audio_uris:
#                 try:
#                     # Save original filename fromm URI:
#                     original_filename = audio_uri.split("/")[-1]
#
#                     # Remove the "file://" prefix if it exists:
#                     if audio_uri.startswith("file://"):
#                         audio_uri = audio_uri.replace("file://", "")
#
#                     # if the audio_uri is a remote url:
#                     if audio_uri.startswith("http://") or audio_uri.startswith(
#                             "https://"):
#                         # Download the audio file:
#                         audio_uri = download_audio_to_temp_file(audio_uri)
#
#                     # if the audio_uri is a data uri:
#                     elif audio_uri.startswith("data:audio/"):
#                         # Write the audio data to a temp file:
#                         audio_uri = write_base64_to_temp_file(audio_uri)
#
#                     if client:
#                         # Open the audio file:
#                         with open(audio_uri, "rb") as audio_file:
#
#                             # Transcribe the audio file:
#                             transcription = client.audio.transcriptions.create(
#                                 model="whisper-1",
#                                 file=audio_file,
#                                 response_format="text"
#                             )
#                     elif is_local_whisper_available():
#                         # We use a local instance of whisper instead:
#                         transcription = transcribe_audio_with_local_whisper(
#                             audio_uri)
#
#                     else:
#                         raise ValueError(
#                             "Whisper is not available. Please install the 'whisper' package to use this feature.")
#
#                     # Add markdown quotes ''' around the transcribed text, and
#                     # add prefix: "Transcription: " to the transcribed text:
#                     transcription = f"\nTranscription of audio file '{original_filename}': \n'''\n{transcription}\n'''\n"
#
#                     # Add the transcribed text to the message
#                     message.append_text(transcription)
#                 except Exception as e:
#                     raise ValueError(
#                         f"Could not transcribe audio '{audio_uri}': {e}")
#
#             # If the audio was transcribed, remove the audio_uris from the message:
#             message.audio_uris = []
#
#     return messages
