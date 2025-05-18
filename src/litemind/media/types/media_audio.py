import pathlib
from typing import Optional

import numpy

from litemind.media.media_uri import MediaURI
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class Audio(MediaURI):
    """
    A media that stores an audio file
    """

    @classmethod
    def from_data(
        cls,
        data: numpy.ndarray,
        sample_rate: int,
        file_format: Optional[str] = None,
        filepath: Optional[str] = None,
    ):
        """
        Create a new audio media from data.
        This method creates a temporary file on disk and returns an Audio object with the URI of that file.
        The file is deleted when the Audio object is deleted.
        The file format is determined by the file extension of the filepath.

        Parameters
        ----------
        data: numpy.ndarray
            The audio data to be saved.
        sample_rate: int
            The sample rate of the audio data.
        file_format: str
            The file format of the audio data. If None, the format is determined from the filepath. If that fails default is WAV.
        filepath: str
            The file path to save the audio data to. If None, a temporary file is created.
            The file format is determined from the file extension of the filepath.

        Returns
        -------
        Audio
            The audio media object.

        """

        # Create sound file from data using soundfile:
        if filepath is None:
            # Create temporary file:
            import tempfile

            filepath = tempfile.NamedTemporaryFile(delete=False).name + ".wav"

        if file_format is None:
            # Get file extension from filepath:
            file_format = "RAW" if filepath.lower().endswith(".raw") else "WAV"

        # Write file to disk:
        import soundfile

        soundfile.write(
            data=data, samplerate=sample_rate, file=filepath, format=file_format
        )

        # Create URI:
        audio_uri = "file://" + filepath

        return Audio(uri=audio_uri)

    def load_from_uri(self):

        # Download the video file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # load the audio file using soundfile:
        import soundfile

        self.data, self.samplerate = soundfile.read(local_file)
        self.num_channels = self.data.shape[1] if len(self.data.shape) > 1 else 1
        self.dtype = self.data.dtype

    def get_raw_data(self) -> bytes:
        # Convert the audio URI to a local file path:
        local_path = uri_to_local_file_path(self.uri)

        # Load the data from the local file:
        raw_data_bytes = pathlib.Path(local_path).read_bytes()

        return raw_data_bytes

    def get_info_markdown(self) -> str:
        """
        Get information about the audio file in markdown format.

        Returns
        -------
        str
            Markdown formatted string containing audio file information.
        """
        try:
            # Ensure data is loaded
            if not hasattr(self, "data") or self.data is None:
                self.load_from_uri()

            # Get local file path
            local_path = uri_to_local_file_path(self.uri)

            # Calculate duration in seconds
            duration = len(self.data) / self.samplerate

            # Format time into minutes and seconds
            minutes = int(duration // 60)
            seconds = duration % 60

            # Get file size
            file_size = pathlib.Path(local_path).stat().st_size
            file_size_mb = file_size / (1024 * 1024)

            # Create markdown string
            markdown = f"""
## Audio Information
- **Filename**: {pathlib.Path(local_path).name}
- **Duration**: {minutes}m {seconds:.2f}s
- **Sample Rate**: {self.samplerate} Hz
- **Channels**: {self.num_channels}
- **Data Type**: {self.dtype}
- **Samples**: {len(self.data)}
- **File Size**: {file_size_mb:.2f} MB
"""

            return markdown
        except Exception as e:
            return f"## Error\nFailed to extract audio information: {str(e)}"
