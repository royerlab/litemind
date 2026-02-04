from functools import lru_cache

from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


@lru_cache()
def is_local_whisper_available() -> bool:
    """
    Check if Whisper is available.

    Whisper requires both the whisper Python package and ffmpeg
    to be installed as a system binary.

    :return: True if Whisper is available, False otherwise.
    """

    try:
        import importlib.util
        import shutil

        # Check if whisper module is available
        if importlib.util.find_spec("whisper") is None:
            return False

        # Check if ffmpeg is available (required by whisper)
        if shutil.which("ffmpeg") is None:
            return False

        return True
    except Exception:
        return False


def transcribe_audio_with_local_whisper(
    audio_uri: str, model_name: str = "turbo"
) -> str:
    """
    Transcribe audio using a local instance of Whisper.

    Parameters
    ----------
    audio_uri : str
        The URI of the audio file to transcribe.
    model_name : str
        The name of the Whisper model to use.
    """

    # Import Whisper here to avoid circular imports
    import whisper

    # Download the audio file if it's a remote URL
    if audio_uri.startswith("http://") or audio_uri.startswith("https://"):
        local_path = uri_to_local_file_path(audio_uri)
    elif audio_uri.startswith("file://"):
        local_path = audio_uri.replace("file://", "")
    else:
        local_path = audio_uri

    # Load the Whisper model
    model = whisper.load_model(model_name)

    # Transcribe the audio file
    result = model.transcribe(audio=local_path)

    return result["text"]
