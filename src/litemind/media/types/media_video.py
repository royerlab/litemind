from typing import List, Optional

from litemind.agent.messages.message_block import MessageBlock
from litemind.media.media_uri import MediaURI
from litemind.utils.ffmpeg_utils import (
    convert_video_to_frames_and_audio,
    get_video_info,
    load_video_as_array,
)
from litemind.utils.file_types.file_extensions import VIDEO_EXTS
from litemind.utils.file_types.file_types import is_video_file
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class Video(MediaURI):
    """
    A media that stores a video
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """
        Create a new video media.

        Parameters
        ----------
        uri: str
            The video URI.
        extension: str
            Extension/Type of the video file in case it is not clear from the URI. This is the extension _without_ the dot -- 'mp4' not '.mp4'.
        kwargs: dict
            Other arguments passed to MediaURI.
        """

        # Check that  the file extension is valid:
        if not is_video_file(uri):
            raise ValueError(
                f"Invalid video URI: '{uri}' (must have a valid video file extension)"
            )

        if extension is not None:
            if "." + extension not in VIDEO_EXTS:
                raise ValueError(f"Invalid video extension: {extension}")

        super().__init__(uri=uri, extension=extension, **kwargs)

        self.array = None
        self.info = None

    def load_from_uri(self):

        if self.array is not None:
            return

        # Download the video file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # Load the video data using ffmpeg:
        array = load_video_as_array(local_file)

        # Store the video data in an attribute:
        self.array = array

        # Get the metadata of the video:
        self.get_video_info()

    def get_video_info(self):
        """
        Get metadata of the video.

        Returns
        -------
        dict
            Metadata of the video.

        """

        if self.info is None:
            # Get the video information using ffmpeg:
            video_info = get_video_info(self.uri)

            # Store the metadata in an attribute:
            self.info = video_info

        return self.info

    def convert_to_frames_and_audio(self) -> List[MessageBlock]:
        """
        Convert the video to frames and audio.
        This method uses ffmpeg to convert the video to frames and audio,
        and returns a list of MessageBlock objects.

        Returns
        -------
        List[MessageBlock]
            A list of MessageBlock objects containing the frames and audio data.

        """

        # Get the local file path for the URI:
        local_path = self.to_local_file_path()

        # Convert the video to frames and audio:
        blocks = convert_video_to_frames_and_audio(local_path)

        return blocks
