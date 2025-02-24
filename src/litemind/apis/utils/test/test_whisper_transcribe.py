import os

from litemind.apis.utils.whisper_transcribe_audio import (
    transcribe_audio_with_local_whisper,
)


def test_transcribe_audio():
    # Get the current directory:
    current_dir = os.path.dirname(__file__)

    # Combine the two to get the absolute path to 'harvard.wav'
    audio_path = os.path.join(current_dir, "media/harvard.wav")

    # Transcribe the audio:
    result = transcribe_audio_with_local_whisper(audio_path)

    print(result)

    # Check if the result is a string and contains the expected words:
    assert isinstance(result, str)
    assert len(result) > 0
    assert "smell" in result or "ham" in result or "beer" in result
