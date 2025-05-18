import os
import tempfile

import numpy as np
import pytest

from litemind.media.types.media_audio import Audio


@pytest.fixture
def sample_audio_data():
    """Fixture providing a simple sine wave audio sample"""
    sample_rate = 44100
    duration = 0.1  # 100ms audio for quick tests
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    return audio_data, sample_rate


@pytest.fixture
def temp_audio_file(sample_audio_data):
    """Fixture creating a temporary audio file"""

    data, sample_rate = sample_audio_data
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    import soundfile as sf

    sf.write(tmp_path, data, sample_rate)

    yield tmp_path

    # Cleanup
    if os.path.exists(tmp_path):
        os.remove(tmp_path)


def test_audio_initialization():
    """Test basic initialization with URI"""
    audio_uri = "file:///path/to/audio.wav"
    audio = Audio(uri=audio_uri)
    assert audio.uri == audio_uri


def test_from_data_default_filepath(sample_audio_data):
    """Test from_data method with auto-generated filepath"""
    data, sample_rate = sample_audio_data
    audio = Audio.from_data(data=data, sample_rate=sample_rate)

    # Verify URI format and that file exists
    assert audio.uri.startswith("file://")
    file_path = audio.uri[7:]  # Remove "file://" prefix
    assert os.path.exists(file_path)

    # Clean up
    os.remove(file_path)


def test_from_data_custom_filepath(sample_audio_data):
    """Test from_data method with custom filepath"""
    data, sample_rate = sample_audio_data

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        custom_path = tmp.name

    try:
        audio = Audio.from_data(
            data=data, sample_rate=sample_rate, filepath=custom_path
        )
        assert audio.uri == f"file://{custom_path}"

        # Verify the data was written correctly
        import soundfile as sf

        loaded_data, loaded_sr = sf.read(custom_path)
        assert loaded_sr == sample_rate
        assert np.allclose(loaded_data, data, atol=1e-3, rtol=1e-3)
    finally:
        if os.path.exists(custom_path):
            os.remove(custom_path)


def test_load_from_uri(temp_audio_file, sample_audio_data):
    """Test loading audio data from URI"""
    original_data, sample_rate = sample_audio_data
    audio = Audio(uri=f"file://{temp_audio_file}")

    # Load the audio data
    audio.load_from_uri()

    # Verify properties
    assert hasattr(audio, "data")
    assert audio.samplerate == sample_rate
    assert audio.num_channels == 1
    assert audio.dtype == original_data.dtype
    assert np.allclose(audio.data, original_data, atol=1e-3, rtol=1e-3)


def test_stereo_audio():
    """Test handling of stereo audio files"""
    # Create stereo audio data
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    channel1 = np.sin(2 * np.pi * 440 * t)
    channel2 = np.sin(2 * np.pi * 880 * t)
    stereo_data = np.column_stack((channel1, channel2))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        stereo_path = tmp.name

    try:
        import soundfile as sf

        sf.write(stereo_path, stereo_data, sample_rate)

        audio = Audio(uri=f"file://{stereo_path}")
        audio.load_from_uri()

        assert audio.num_channels == 2
        assert audio.data.shape == stereo_data.shape
        assert np.allclose(audio.data, stereo_data, atol=1e-3, rtol=1e-3)
    finally:
        if os.path.exists(stereo_path):
            os.remove(stereo_path)


def test_different_audio_formats():
    """Test with different audio formats (WAV, FLAC)"""
    sample_rate = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = np.sin(2 * np.pi * 440 * t)

    for fmt, suffix in [("WAV", ".wav"), ("FLAC", ".flac")]:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            format_path = tmp.name

        try:
            import soundfile as sf

            sf.write(format_path, audio_data, sample_rate, format=fmt)

            audio = Audio(uri=f"file://{format_path}")
            audio.load_from_uri()

            assert audio.samplerate == sample_rate
            assert np.allclose(audio.data, audio_data, atol=1e-3, rtol=1e-3)
        finally:
            if os.path.exists(format_path):
                os.remove(format_path)


def test_from_data_non_array_input():
    """Test from_data with invalid input types"""
    with pytest.raises(Exception):
        Audio.from_data(data="not an array", sample_rate=44100)
