"""N-dimensional image converter for NdImage media.

Provides :class:`NdImageConverter` which converts NdImage media into Text
descriptions and 2-D Image projections suitable for LLM consumption.
"""

from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_image import Image
from litemind.media.types.media_ndimage import NdImage
from litemind.media.types.media_text import Text


class NdImageConverter(BaseConverter):
    """Converts NdImage media to Text descriptions and 2-D Image projections."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms NdImage to Text and Image.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping NdImage to Text and Image.
        """
        return [(NdImage, [Text, Image])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is an NdImage instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a non-None NdImage instance.
        """
        return media is not None and isinstance(media, NdImage)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert NdImage media to Text descriptions and 2-D Image projections.

        Parameters
        ----------
        media : MediaBase
            The NdImage media to convert.

        Returns
        -------
        List[MediaBase]
            A list containing Text metadata and Image projections of the
            n-dimensional image.

        Raises
        ------
        ValueError
            If *media* is not an NdImage instance.
        """
        if not isinstance(media, NdImage):
            raise ValueError(f"Expected NdImage media, got {type(media)}")

        # Use the NdImage's built-in method to convert to Text and Images:
        return media.to_text_and_2d_projection_medias()
