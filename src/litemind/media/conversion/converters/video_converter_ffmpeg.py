import os
import tempfile
from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.utils.ffmpeg_utils import (
    extract_frames_and_audio,
    get_video_info,
    is_ffmpeg_available,
)
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class VideoConverterFfmpeg(BaseConverter):
    """
    Converter for Video media type.

    Converts Video media to Text, Image and Audio media.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(Video, [Text, Image, Audio])]

    def can_convert(self, media: MediaBase) -> bool:
        return media is not None and isinstance(media, Video)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        if not isinstance(media, Video):
            raise ValueError(f"Expected Video media, got {type(media)}")

        # Convert the video to info, frames and audio:
        media_list = convert_video_to_info_frames_and_audio(media)

        return media_list


def convert_video_to_info_frames_and_audio(
    media: MediaBase,
    frame_interval: int = 1,
    key_frames: bool = False,
) -> List[MediaBase]:
    """
    Converts a video media into image frames, an audio file, and video info.

    Parameters
    ----------
    media: MediaBase
        Media to convert.
    frame_interval: int
        Interval in seconds to sample frames from the video.
    key_frames: bool
        Whether to extract key frames instead of sampling at regular intervals.

    Returns
    -------
    List[MediaBase]
        The corresponding audio, images and text media describing the video
    """

    # Check if ffmpeg is available:
    if not is_ffmpeg_available():
        raise RuntimeError("FFmpeg is not available. Please install FFmpeg.")

    # Check that the media is a Video:
    if not isinstance(media, Video):
        raise ValueError(f"Expected Video media, got {type(media)}")

    # Get the media URI:
    media_uri = media.uri

    # Convert the document URI to a local file path.
    video_path = uri_to_local_file_path(media_uri)

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

    # Initialise an empty string for the metadata:
    metadata_string = ""

    # Append video filename:
    metadata_string += f"Video: {os.path.basename(video_path)}\n"

    # Append video information to the message
    metadata_string += f"Video duration: {video_info['duration']} seconds.\n"
    metadata_string += f"Video resolution: {video_info['resolution'][0]}x{video_info['resolution'][1]}.\n"
    metadata_string += f"Video codec: {video_info['codec']}.\n"
    metadata_string += f"Video bit rate: {video_info['bit_rate']} bps.\n"
    metadata_string += f"Video frame rate: {video_info['frame_rate']} fps.\n"
    metadata_string += (
        f"Frames sampled every {frame_interval} seconds.\n"
        if not key_frames
        else "Key frames extracted.\n"
    )

    # Total number of frames:
    total_frames = len(frames)
    metadata_string += f"the video content is provided below as a sequence of {total_frames} image frames:\n"

    media_list = []
    media_list.append(Text(metadata_string))

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
        media_list.append(Text(f"Frame at {time_str} ({i}/ {total_frames}): "))
        media_list.append(Image("file://" + frame_path))

    # Append audio to the message if it exists:
    if audio_path is not None:
        media_list.append(Text(f"The video's audio is provided separately below:\n"))
        # Append audio to the message
        media_list.append(Audio("file://" + audio_path))

    return media_list
