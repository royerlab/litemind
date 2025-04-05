import numpy as np
import pytest

from litemind.media.types.media_video import Video
from litemind.ressources.media_resources import MediaResources
from litemind.utils.file_extensions import video_file_extensions


class TestVideo:
    def test_init_with_valid_uri(self):
        """Test initializing Video with a valid URI."""
        # Get a real video URI from MediaResources
        video_uri = MediaResources.get_local_test_video_uri("flying.mp4")
        video = Video(uri=video_uri)
        assert video.uri == video_uri

    def test_init_with_invalid_uri(self):
        """Test that initializing with invalid URI raises ValueError."""
        invalid_uri = MediaResources.get_local_test_image_uri("future.jpeg")
        with pytest.raises(ValueError, match="Invalid video URI"):
            Video(uri=invalid_uri)

    def test_init_with_valid_extension(self):
        """Test initializing Video with valid extension parameter."""
        video_uri = MediaResources.get_local_test_video_uri("bunny.mp4")
        video = Video(uri=video_uri, extension="mp4")
        assert video.uri == video_uri

    def test_init_with_invalid_extension(self):
        """Test that initializing with invalid extension raises ValueError."""
        video_uri = MediaResources.get_local_test_video_uri("lunar_park.mp4")
        with pytest.raises(ValueError, match="Invalid video extension"):
            Video(uri=video_uri, extension="invalid")

    def test_load_from_uri(self):
        """Test the load_from_uri method with a real video file."""
        video_uri = MediaResources.get_local_test_video_uri("flying.mp4")
        video = Video(uri=video_uri)
        video.load_from_uri()

        # Verify videodata is loaded
        assert hasattr(video, "array")
        assert isinstance(video.array, np.ndarray)
        assert video.array.ndim == 4  # time, height, width, channels

        # Verify metadata is loaded
        assert hasattr(video, "info")
        assert isinstance(video.info, dict)
        assert (
            "bit_rate" in video.info
            and "codec" in video.info
            and "duration" in video.info
            and "resolution" in video.info
        )

    def test_get_metadata(self):
        """Test that get_metadata returns the correct metadata."""
        video_uri = MediaResources.get_local_test_video_uri("bunny.mp4")
        video = Video(uri=video_uri)

        # Get metadata
        info = video.get_video_info()

        # Verify returned metadata
        assert isinstance(info, dict)
        assert (
            "bit_rate" in video.info
            and "codec" in video.info
            and "duration" in video.info
            and "resolution" in video.info
        )

    def test_video_properties(self):
        """Test various properties of the loaded video."""
        video_uri = MediaResources.get_local_test_video_uri("lunar_park.mov")
        video = Video(uri=video_uri)
        video.load_from_uri()

        # Check dimensions
        assert video.array.shape[0] > 0  # At least 1 frame
        assert video.array.shape[1] > 0  # Height
        assert video.array.shape[2] > 0  # Width
        assert video.array.shape[3] == 3  # RGB channels

        # Check metadata contains useful info
        info = video.get_video_info()
        assert (
            "bit_rate" in info
            and "codec" in info
            and "duration" in info
            and "resolution" in info
        )

    def test_file_extensions(self):
        """Test that the video file extensions are properly validated."""
        for ext in video_file_extensions:
            fake_uri = f"file:///path/to/video.{ext}"
            # This should not raise an exception
            Video(uri=fake_uri)
