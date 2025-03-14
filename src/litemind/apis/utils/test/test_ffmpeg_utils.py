import os

import pytest

from litemind.apis.utils.ffmpeg_utils import (
    extract_frames_and_audio,
    is_ffmpeg_available,
)


@pytest.fixture
def test_video(tmp_path):
    """
    Generate a short 2-second synthetic tests video (with audio)
    and return the path to the created file.
    """

    import ffmpeg

    # Paths for the generated video and final tests file
    video_path = tmp_path / "test_synthetic_video.mp4"

    # Create synthetic video input (red color)
    v = ffmpeg.input("color=c=red:size=320x240:rate=30:d=2", f="lavfi")
    # Create synthetic audio input (440Hz sine wave)
    a = ffmpeg.input("sine=frequency=440:duration=2", f="lavfi")

    # Combine video + audio into one MP4 file
    (
        ffmpeg.output(
            v,
            a,
            str(video_path),
            vcodec="libx264",
            acodec="aac",  # or 'libmp3lame' if needed
            strict="experimental",  # may be needed for aac
            pix_fmt="yuv420p",
            r=30,  # frames per second
        )
        .overwrite_output()
        .run()
    )

    return str(video_path)


def test_is_ffmpeg_available():
    """
    Test if ffmpeg is available.
    """

    # Check that the return of is_ffmpeg_available() is boolean:
    assert isinstance(
        is_ffmpeg_available(), bool
    ), "is_ffmpeg_available() should return a boolean."


def test_extract_frames_and_audio(tmp_path, test_video):
    """
    Test that frames and audio are correctly extracted from the video file.
    """
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    # Call the function under tests
    frames, audio_file = extract_frames_and_audio(
        input_video_path=test_video,
        output_dir=str(output_dir),
        fps=1.0,  # extract 1 frame per second
        use_keyframes=False,
        audio_filename="audio.wav",
        audio_sample_rate=16000,
        audio_channels=1,
    )

    # --- Check results ---
    # 1) Frames
    # We expect about 2 frames (since it's a 2s video at 1 FPS),
    # but the exact number can vary by an off-by-one depending on how
    # ffmpeg truncates or rounds the duration. Let's check we have > 0 frames:
    assert len(frames) > 0, "No frames were extracted."

    # Check that frames exist on disk
    for frame in frames:
        assert os.path.isfile(frame), f"Frame file {frame} does not exist."
        assert os.path.getsize(frame) > 0, f"Frame file {frame} is empty."

    # 2) Audio file
    assert os.path.isfile(audio_file), "No audio file was extracted."
    # Check that the extracted audio file is non-empty.
    assert os.path.getsize(audio_file) > 0, "Extracted audio file is empty."

    # Validate the length or sampling rate of the generated audio
    import wave

    with wave.open(audio_file, "r") as audio:
        assert audio.getnchannels() == 1, "Audio file should have 1 channel."
        assert (
            audio.getframerate() == 16000
        ), "Audio file should have a sample rate of 16000 Hz."

    # (Optional) Print for debugging or demonstration
    print("Extracted frames:", frames)
    print("Extracted audio file:", audio_file)
