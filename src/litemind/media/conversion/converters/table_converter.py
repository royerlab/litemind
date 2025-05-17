from typing import List

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_table import Table


class TableConverter(BaseConverter):
    """
    Converter for Table media type.

    Converts Table media to Text media.
    """

    def can_convert(self, media: MediaBase) -> bool:

        return media is not None and isinstance(media, Table)

    def convert(self, media: MediaBase) -> List[MediaBase]:

        if not isinstance(media, Table):
            raise ValueError(f"Expected Table media, got {type(media)}")

        # Use the Table's built-in method to convert to Text
        return [media.to_markdown_text_media()]