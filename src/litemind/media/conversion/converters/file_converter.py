"""Generic file-to-text converter for File media.

Provides :class:`FileConverter` which converts File media into Text by
rendering metadata and content preview as Markdown. This is a low-priority
catch-all converter for files not handled by more specific converters.
"""

from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_file import File
from litemind.media.types.media_text import Text


class FileConverter(BaseConverter):
    """Converts generic File media to Text with metadata and content preview.

    This is a low-priority catch-all converter for files not handled by
    more specific converters.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms File to Text.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping File to Text.
        """
        return [(File, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is a File instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a non-None File instance.
        """
        return media is not None and isinstance(media, File)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert File media to a Markdown-formatted Text media.

        Uses the File's built-in ``to_markdown_text_media()`` method to
        generate a text representation with metadata and content preview.

        Parameters
        ----------
        media : MediaBase
            The File media to convert.

        Returns
        -------
        List[MediaBase]
            A single-element list containing the Text representation.

        Raises
        ------
        ValueError
            If *media* is not a File instance.
        """
        if not isinstance(media, File):
            raise ValueError(f"Expected File media, got {type(media)}")

        file: File = media

        # Use the File's built-in method to convert to Text
        return [file.to_markdown_text_media()]
