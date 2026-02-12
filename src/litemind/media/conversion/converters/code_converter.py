"""Code-to-text converter for rendering Code media as Markdown code blocks.

Provides :class:`CodeConverter` which transforms Code media into Text media
containing a Markdown-formatted fenced code block.
"""

from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_code import Code
from litemind.media.types.media_text import Text


class CodeConverter(BaseConverter):
    """Converts Code media to Text by rendering as a Markdown code block."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Code to Text.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Code to Text.
        """
        return [(Code, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is a Code instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a non-None Code instance.
        """
        return media is not None and isinstance(media, Code)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert Code media to a Markdown-formatted Text media.

        Parameters
        ----------
        media : MediaBase
            The Code media to convert.

        Returns
        -------
        List[MediaBase]
            A single-element list containing the Text representation.

        Raises
        ------
        ValueError
            If *media* is not a Code instance.
        """
        if not isinstance(media, Code):
            raise ValueError(f"Expected Code media, got {type(media)}")

        # Use the Code's built-in method to convert to Text
        return [media.to_markdown_text_media()]
