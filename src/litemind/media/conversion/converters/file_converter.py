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
        return [(File, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        return media is not None and isinstance(media, File)

    def convert(self, media: MediaBase) -> List[MediaBase]:
        if not isinstance(media, File):
            raise ValueError(f"Expected File media, got {type(media)}")

        file: File = media

        # Use the File's built-in method to convert to Text
        return [file.to_markdown_text_media()]
