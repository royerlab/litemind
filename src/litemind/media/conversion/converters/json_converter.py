"""JSON-to-text converter for rendering Json media as Markdown code blocks.

Provides :class:`JsonConverter` which transforms Json media into Text media
containing a Markdown-formatted JSON code block.
"""

from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_json import Json
from litemind.media.types.media_text import Text


class JsonConverter(BaseConverter):
    """Converts Json media to Text by rendering as a Markdown JSON code block."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Json to Text.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Json to Text.
        """
        return [(Json, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is a Json instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a non-None Json instance.
        """
        return media is not None and isinstance(media, Json)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert Json media to a Markdown-formatted Text media.

        Parameters
        ----------
        media : MediaBase
            The Json media to convert.

        Returns
        -------
        List[MediaBase]
            A single-element list containing the Text representation.

        Raises
        ------
        ValueError
            If *media* is not a Json instance.
        """
        if not isinstance(media, Json):
            raise ValueError(f"Expected Json media, got {type(media)}")

        # Use the Json's built-in method to convert to Text
        return [media.to_markdown_text_media()]
