"""Object-to-text converter for Pydantic BaseModel media.

Provides :class:`ObjectConverter` which converts Object media (wrapping
Pydantic BaseModel instances) into Text media containing a Markdown-formatted
JSON representation.
"""

from typing import Any, List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_object import Object
from litemind.media.types.media_text import Text


class ObjectConverter(BaseConverter):
    """Converts Object (Pydantic BaseModel) media to Text via JSON Markdown."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Object to Text.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Object to Text.
        """
        return [(Object, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is an Object instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a non-None Object instance.
        """
        return media is not None and isinstance(media, Object)

    def convert(self, media: Any) -> List[MediaBase]:
        """Convert Object media to a Markdown JSON Text media.

        Parameters
        ----------
        media : Any
            The Object media to convert.

        Returns
        -------
        List[MediaBase]
            A single-element list containing the Text representation.

        Raises
        ------
        ValueError
            If *media* is not an Object instance.
        """
        if not isinstance(media, Object):
            raise ValueError(f"Expected Object media, got {type(media)}")

        # Use the Object's built-in method to convert to Text
        return [media.to_markdown_text_media()]
