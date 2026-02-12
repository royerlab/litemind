"""Fallback document-to-text converter for text-based files.

Provides :class:`DocumentConverterTxt` which reads plain text, code, and
script files directly and wraps them in a Markdown fenced code block.
This converter is intended as a low-priority fallback.
"""

from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_document import Document
from litemind.media.types.media_text import Text
from litemind.utils.file_types.file_types import classify_uri
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class DocumentConverterTxt(BaseConverter):
    """Converts text-based documents (plain text, code, scripts) to Text media.

    This is a low-priority fallback converter that reads the file content
    directly and wraps it in a Markdown fenced code block with the original
    filename and extension.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Document to Text.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Document to Text.
        """
        return [(Document, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is a text-based Document.

        Supports documents classified as ``text``, ``code``, or ``script``.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a Document with a text-based file type.
        """
        # Check is the media is None:
        if media is None:
            return False

        # Check if the media is a Document:
        if not isinstance(media, Document):
            return False

        # Check if the file is a text-based document:
        file_type = classify_uri(media.uri)
        if file_type not in {"text", "code", "script"}:
            return False

        # By default, we can convert the media, unless the previous checks failed:
        return True

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Read a text-based Document and wrap its content in a Markdown code block.

        Parameters
        ----------
        media : MediaBase
            The Document media to convert.

        Returns
        -------
        List[MediaBase]
            A single-element list containing a Text media with the file
            content wrapped in a Markdown fenced code block.

        Raises
        ------
        ValueError
            If *media* is not a Document instance.
        """
        if not isinstance(media, Document):
            raise ValueError(f"Expected Document media, got {type(media)}")

        # Normalise uri to local path:
        document_path = uri_to_local_file_path(media.uri)

        # get filename:
        filename = media.get_filename()

        # Get the file extension:
        extension = media.get_extension()

        # Read the content of the text document from file:
        with open(document_path, "r", encoding="utf-8") as file:
            text_content = file.read()

        # Wrap file in markdown quotes and give filename:
        text_content = f"{filename}\n```{extension}\n{text_content}\n```"

        # Create a Text media object:
        text_media = Text(text_content)

        # Return the converted media:
        return [text_media]
