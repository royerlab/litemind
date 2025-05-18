from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_image import Image
from litemind.media.types.media_ndimage import NdImage
from litemind.media.types.media_text import Text


class NdImageConverter(BaseConverter):
    """
    Converter for File media type.

    Converts File media to Text media, this is for file media that are not any of the other known media.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(NdImage, [Text, Image])]

    def can_convert(self, media: MediaBase) -> bool:
        return media is not None and isinstance(media, NdImage)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        if not isinstance(media, NdImage):
            raise ValueError(f"Expected NdImage media, got {type(media)}")

        # Use the NdImage's built-in method to convert to Text and Images:
        return media.to_text_and_2d_projection_medias()
