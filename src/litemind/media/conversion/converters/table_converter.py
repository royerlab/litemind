"""Table-to-text converter for rendering Table media as Markdown tables.

Provides :class:`TableConverter` which transforms Table media into Text media
containing a Markdown-formatted table representation.
"""

from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text


class TableConverter(BaseConverter):
    """Converts Table media to Text by rendering as a Markdown table."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Table to Text.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Table to Text.
        """
        return [(Table, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is a Table instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a non-None Table instance.
        """
        return media is not None and isinstance(media, Table)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert Table media to a Markdown-formatted Text media.

        Parameters
        ----------
        media : MediaBase
            The Table media to convert.

        Returns
        -------
        List[MediaBase]
            A single-element list containing the Text representation.

        Raises
        ------
        ValueError
            If *media* is not a Table instance.
        """
        if not isinstance(media, Table):
            raise ValueError(f"Expected Table media, got {type(media)}")

        # Use the Table's built-in method to convert to Text
        return [media.to_markdown_text_media()]
