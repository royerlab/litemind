"""Video-to-frames-and-audio converter using ffmpeg.

Provides :class:`VideoConverterFfmpeg` which decomposes Video media into
Text metadata, sampled Image frames, and an optional Audio track using
the ffmpeg command-line tool.
"""

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
    """Converts Video media to Text metadata, Image frames, and Audio track.

    Requires ffmpeg to be installed and accessible on the system PATH.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Video to Text and Image.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Video to Text and Image.
        """
        return [(Video, [Text, Image])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is a Video instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a non-None Video instance.
        """
        return media is not None and isinstance(media, Video)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert Video media to metadata Text, Image frames, and Audio.

        Delegates to :func:`convert_video_to_info_frames_and_audio` to
        decompose the video into its constituent parts.

        Parameters
        ----------
        media : MediaBase
            The Video media to convert.

        Returns
        -------
        List[MediaBase]
            A list containing a Text summary, alternating Text/Image pairs
            for each sampled frame, and optionally an Audio track.

        Raises
        ------
        ValueError
            If *media* is not a Video instance.
        """
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
    """Decompose a video into metadata text, image frames, and an audio track.

    Parameters
    ----------
    media : Video
        The video media to decompose.
    frame_interval : int, optional
        Interval in seconds between sampled frames. Default is 1.
    key_frames : bool, optional
        If True, extract key frames instead of sampling at regular
        intervals. Default is False.

    Returns
    -------
    List[MediaBase]
        A list starting with a Text summary, followed by alternating
        Text/Image pairs for each frame, and optionally an Audio media
        for the soundtrack.

    Raises
    ------
    RuntimeError
        If ffmpeg is not available.
    ValueError
        If *media* is not a Video instance.
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
    from litemind.utils.temp_file_manager import register_temp_dir

    output_dir = register_temp_dir(tempfile.mkdtemp())

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
        media_list.append(Text("The video's audio is provided separately below:\n"))
        # Append audio to the message
        media_list.append(Audio("file://" + audio_path))

    return media_list
