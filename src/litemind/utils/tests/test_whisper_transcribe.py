import os

import pytest

from litemind.ressources.media_resources import MediaResources
from litemind.utils.whisper_transcribe_audio import (
    is_local_whisper_available,
    transcribe_audio_with_local_whisper,
)


def test_transcribe_audio():
    # If local whisper is not available, skip the test:
    if not is_local_whisper_available():
        pytest.skip("Local Whisper is not available. Skipping test.")

    # Get the current directory:
    current_dir = os.path.dirname(__file__)

    # Combine the two to get the absolute path to 'harvard.wav'
    audio_path = MediaResources.get_local_test_audio_uri("harvard.wav")

    # Transcribe the audio:
    result = transcribe_audio_with_local_whisper(audio_path)

    print(result)

    # Check if the result is a string and contains the expected words:
    assert isinstance(result, str)
    assert len(result) > 0
    assert "smell" in result or "ham" in result or "beer" in result
