import pathlib
from functools import lru_cache
from typing import Optional

import numpy

from litemind.media.media_uri import MediaURI
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


@lru_cache()
def is_soundfile_available() -> bool:
    """Check whether the ``soundfile`` library is installed.

    Returns
    -------
    bool
        True if ``soundfile`` can be imported, False otherwise.
    """
    try:
        import soundfile  # noqa: F401

        return True
    except Exception:
        return False


class Audio(MediaURI):
    """Media that stores an audio file referenced by URI.

    Audio data can be created from numpy arrays via ``from_data`` or
    loaded from an existing file. The ``soundfile`` library is used for
    reading and writing audio data.
    """

    @classmethod
    def from_data(
        cls,
        data: numpy.ndarray,
        sample_rate: int,
        file_format: Optional[str] = None,
        filepath: Optional[str] = None,
    ):
        """Create an Audio media from a numpy array.

        The array is written to disk using ``soundfile`` and wrapped in an
        Audio media instance.

        Parameters
        ----------
        data : numpy.ndarray
            The audio sample data (1-D for mono, 2-D for multi-channel).
        sample_rate : int
            The sample rate in Hz.
        file_format : str, optional
            The audio file format (e.g. ``"WAV"``, ``"RAW"``). If None,
            the format is inferred from *filepath* or defaults to WAV.
        filepath : str, optional
            Destination file path. A temporary WAV file is created if None.

        Returns
        -------
        Audio
            A new Audio media referencing the saved file.
        """

        # Create sound file from data using soundfile:
        if filepath is None:
            # Create temporary file:
            import tempfile

            from litemind.utils.temp_file_manager import register_temp_file

            filepath = register_temp_file(
                tempfile.NamedTemporaryFile(delete=False).name + ".wav"
            )

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
        """Load audio data from the URI into instance attributes.

        Populates ``self.data`` (numpy array), ``self.samplerate`` (int),
        ``self.num_channels`` (int), and ``self.dtype``.
        """

        # Download the audio file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # load the audio file using soundfile:
        import soundfile

        self.data, self.samplerate = soundfile.read(local_file)
        self.num_channels = self.data.shape[1] if len(self.data.shape) > 1 else 1
        self.dtype = self.data.dtype

    def get_raw_data(self) -> bytes:
        """Read the raw bytes of the audio file.

        Returns
        -------
        bytes
            The unprocessed file content.
        """
        # Convert the audio URI to a local file path:
        local_path = uri_to_local_file_path(self.uri)

        # Load the data from the local file:
        raw_data_bytes = pathlib.Path(local_path).read_bytes()

        return raw_data_bytes

    def get_info_markdown(self) -> str:
        """Get audio file metadata formatted as Markdown.

        Loads the audio data if not already loaded, then returns a Markdown
        string with filename, duration, sample rate, channels, data type,
        sample count, and file size.

        Returns
        -------
        str
            Markdown-formatted audio information, or an error message if
            metadata extraction fails.
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
