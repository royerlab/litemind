import os

# _set_ffmpeg_binary()
import subprocess
import tempfile
from functools import lru_cache
from typing import List, Optional

from arbol import aprint
from numpy import array, frombuffer, uint8

from litemind.agent.messages.message_block import MessageBlock


# function that checks if ffmpeg is available and functional:
@lru_cache()
def is_ffmpeg_available():
    try:
        import importlib.util

        if importlib.util.find_spec("ffmpeg") is None:
            aprint("FFmpeg is not available.")
            return False

        process = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if process.returncode == 0:
            aprint("FFmpeg is available.")
            # aprint(process.stdout)
            return True
        else:
            aprint("FFmpeg is not available.")
            # aprint(process.stderr)
            return False
    except FileNotFoundError:
        aprint("FFmpeg is not found in the system's PATH.")
        return False


def convert_video_to_frames_and_audio(
    video_path: str,
    message: Optional["Message"] = None,
    frame_interval: int = 1,
    key_frames: bool = False,
) -> List[MessageBlock]:
    """
    Converts a video into message blocks by splitting it into frames and an audio file, and appending the frames, audio, and video info.

    Parameters
    ----------
    video_path: str
        Path to the video file.
    message: Optional[Message]
        The existing Message object to append the video information to. If not provided an empty message is used.
    frame_interval: int
        Interval in seconds to sample frames from the video.
    key_frames: bool
        Whether to extract key frames instead of sampling at regular intervals.

    Returns
    -------
    List[MessageBlock]
        The corresponding message blocks for the audio, images and text describing the video
    """

    # define a temporary folder:
    output_dir = tempfile.mkdtemp()

    # Extract frames and audio
    frames, audio_path = extract_frames_and_audio(
        input_video_path=video_path,
        output_dir=output_dir,
        fps=frame_interval,
        use_keyframes=key_frames,
    )

    # Get video information
    video_info = get_video_info(video_path)

    # if the message is not provided, initialise an empty message:
    if message is None:
        from litemind.agent.messages.message import Message

        message = Message()

    # Append video filename:
    message.append_text(f"Video: {os.path.basename(video_path)}\n")

    # Append video information to the message
    message.append_text(f"Video duration: {video_info['duration']} seconds.\n")
    message.append_text(
        f"Video resolution: {video_info['resolution'][0]}x{video_info['resolution'][1]}.\n"
    )
    message.append_text(f"Video codec: {video_info['codec']}.\n")
    message.append_text(f"Video bit rate: {video_info['bit_rate']} bps.\n")
    message.append_text(f"Video frame rate: {video_info['frame_rate']} fps.\n")
    message.append_text(
        f"Frames sampled every {frame_interval} seconds.\n"
        if not key_frames
        else "Key frames extracted.\n"
    )

    # Total number of frames:
    total_frames = len(frames)
    message.append_text(
        f"the video content is provided as a sequence of image frames.\n"
    )
    message.append_text(f"Total frames: {total_frames}.\n")

    # Append frames to the message
    for i, frame_path in enumerate(frames):
        # estimate time in seconds:
        time = i * frame_interval

        # Format time into hours, minutes, and seconds:
        hours = int(time // 3600)
        minutes = int((time % 3600) // 60)
        seconds = int(time % 60)
        time_str = f"{hours:02d}hr {minutes:02d}min {seconds:02d}sec"

        # Append frame to the message
        message.append_text(f"Frame at {time_str} ({i}/ {total_frames}).\n")
        message.append_image("file://" + frame_path)

    # Append audio to the message if it exists:
    if audio_path is not None:
        message.append_text(
            f"The video's audio is provided as a separate audio file.\n"
        )
        # Append audio to the message
        message.append_audio("file://" + audio_path)

    return message.blocks


def get_video_info(video_path: str):
    """
    Get the duration, resolution, codec, bit rate, and frame rate of a video using ffmpeg.

    Parameters
    ----------
    video_path: str
        Path to the video file.

    Returns
    -------
    dict
        Dictionary containing video information.
    """
    import ffmpeg

    probe = ffmpeg.probe(video_path)
    video_info = next(
        stream for stream in probe["streams"] if stream["codec_type"] == "video"
    )

    duration = float(video_info["duration"])
    width = int(video_info["width"])
    height = int(video_info["height"])
    codec = video_info["codec_name"]
    bit_rate = int(video_info["bit_rate"])
    frame_rate = eval(video_info["r_frame_rate"])  # Convert frame rate to float

    return {
        "duration": duration,
        "resolution": (width, height),
        "codec": codec,
        "bit_rate": bit_rate,
        "frame_rate": frame_rate,
    }


def extract_frames_and_audio(
    input_video_path: str,
    output_dir: str,
    image_format: str = "png",
    fps: float = 1.0,
    use_keyframes: bool = False,
    audio_filename: str = "audio.wav",
    audio_sample_rate: int = 16000,
    audio_channels: int = 1,
):
    """
    Extract frames from a video at the specified FPS, or optionally only keyframes.
    Also extract the audio track as a mono WAV file at the given sample rate.

    Args:
        input_video_path (str): Path to the input video file.
        output_dir (str): Directory where the frames and audio will be written.
        image_format (str, optional): The format to use for the extracted frames. Defaults to 'png'.
        fps (float, optional): The frame rate at which to sample video. Defaults to 1.0.
            Ignored if use_keyframes is True.
        use_keyframes (bool, optional): If True, extract only keyframes. Defaults to False.
        audio_filename (str, optional): Name of the output WAV file. Defaults to 'audio.wav'.
        audio_sample_rate (int, optional): The sampling rate for the output WAV file. Defaults to 16000.
        audio_channels (int, optional): The number of channels for the output WAV file. Defaults to 1.

    Returns:
        tuple:
            - list of str: Sorted list of paths to the extracted frame images.
            - str: Path to the extracted audio WAV file.
    """

    # Import ffmpeg
    import ffmpeg

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate pattern for output frames
    frame_pattern = os.path.join(output_dir, f"frame_%06d.{image_format}")

    # Extract frames
    # If use_keyframes is True, select only I-frames (a.k_dict.a. keyframes).
    if use_keyframes:
        # In some cases, you can add `eq(pict_type,I)*eq(key,1)` or `eq(pict_type\, I)` as a filter.
        # `vsync='vfr'` helps preserve variable frame rates correctly when using 'select'.
        (
            ffmpeg.input(input_video_path)
            .filter("select", "eq(pict_type,I)")
            .output(
                frame_pattern, vsync="vfr", qscale=2
            )  # qscale=2 for decent JPEG quality
            .global_args("-loglevel", "quiet", "-nostats")  # Suppress stderr output
            .run(quiet=True, overwrite_output=True)
        )
    else:
        (
            ffmpeg.input(input_video_path)
            .filter("fps", fps=fps)
            .output(frame_pattern, qscale=2)  # qscale=2 for decent JPEG quality
            .global_args("-loglevel", "quiet", "-nostats")  # Suppress stderr output
            .run(quiet=True, overwrite_output=True)
        )

    # First determine if there exists an audio stream:
    probe = ffmpeg.probe(input_video_path)
    audio_streams = [
        stream for stream in probe["streams"] if stream["codec_type"] == "audio"
    ]

    # Extract audio to WAV if an audio stream exists:
    if audio_streams:

        # Extract audio to a WAV file
        audio_output_path = os.path.join(output_dir, audio_filename)
        (
            ffmpeg.input(input_video_path)
            .output(
                audio_output_path,
                format="wav",
                acodec="pcm_s16le",  # 16-bit PCM
                ac=audio_channels,  # number of channels (mono = 1)
                ar=audio_sample_rate,  # sample rate
            )
            .global_args("-loglevel", "quiet", "-nostats")  # Suppress stderr output
            .run(quiet=True, overwrite_output=True)
        )
    else:
        audio_output_path = None

    # Collect all frame filenames (which may be .jpg, .jpeg, or .png if you adapt)
    extracted_frames = sorted(
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith(f".{image_format}")
    )

    return extracted_frames, audio_output_path


def load_video_as_array(filename):
    import ffmpeg

    try:
        probe = ffmpeg.probe(filename)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])

        process = (
            ffmpeg.input(filename)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run_async(pipe_stdout=True)
        )

        video_array = []
        while True:
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                break
            frame = frombuffer(in_bytes, uint8).reshape([height, width, 3])
            video_array.append(frame)
        process.wait()
        return array(video_array)
    except ffmpeg.Error as e:
        aprint(f"FFmpeg error: {e.stderr.decode()}")
        return None
