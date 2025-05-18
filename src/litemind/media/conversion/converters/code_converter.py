from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_code import Code
from litemind.media.types.media_text import Text


class CodeConverter(BaseConverter):
    """
    Converter for Code media type.

    Converts Code media to Text media.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(Code, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        return media is not None and isinstance(media, Code)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        if not isinstance(media, Code):
            raise ValueError(f"Expected Json media, got {type(media)}")

        # Use the Table's built-in method to convert to Text
        return [media.to_markdown_text_media()]
