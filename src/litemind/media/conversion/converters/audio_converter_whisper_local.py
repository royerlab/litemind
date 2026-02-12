"""Audio-to-text converter using a locally installed Whisper model.

Provides :class:`AudioConverterWhisperLocal` which transcribes audio files
to text using OpenAI's Whisper speech recognition model running locally.
"""

from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_text import Text
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path
from litemind.utils.whisper_transcribe_audio import (
    is_local_whisper_available,
    transcribe_audio_with_local_whisper,
)


class AudioConverterWhisperLocal(BaseConverter):
    """Converts Audio media to Text using a local Whisper model.

    Produces a Text media containing audio metadata and the transcription.
    Requires the Whisper library to be installed locally.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Audio to Text.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Audio to Text.
        """
        return [(Audio, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is an Audio instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a non-None Audio instance.
        """
        return media is not None and isinstance(media, Audio)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Transcribe audio to text using a local Whisper model.

        Parameters
        ----------
        media : MediaBase
            The Audio media to transcribe.

        Returns
        -------
        List[MediaBase]
            A single-element list containing a Text media with audio
            metadata and the transcription.

        Raises
        ------
        ValueError
            If *media* is not an Audio instance.
        RuntimeError
            If the local Whisper model is not available.
        """
        if not isinstance(media, Audio):
            raise ValueError(f"Expected Audio media, got {type(media)}")

        # Check if Whisper is available:
        if not is_local_whisper_available():
            raise RuntimeError(
                "Whisper is not available. Please install Whisper to use this converter."
            )

        # Get the media URI:
        media_uri = media.uri

        # Convert the document URI to a local file path.
        audio_path = uri_to_local_file_path(media_uri)

        # Get metadata from the audio file:
        audio_info = media.get_info_markdown()

        # Transcribe the audio file:
        transcription = transcribe_audio_with_local_whisper(audio_path)

        media_list = []

        media_list.append(
            Text(
                audio_info
                + "\n\nThe following is the transcription of the audio:\n"
                + transcription
            )
        )

        return media_list
