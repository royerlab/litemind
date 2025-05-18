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
    """
    Converter for Audio media type.
    Converts Audio media to Text media.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(Audio, [Text])]

    def can_convert(self, media: MediaBase) -> bool:

        return media is not None and isinstance(media, Audio)

    def convert(self, media: MediaBase) -> List[MediaBase]:

        if not isinstance(media, Audio):
            raise ValueError(f"Expected Video media, got {type(media)}")

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
