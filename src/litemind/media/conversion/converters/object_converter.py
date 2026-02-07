from typing import Any, List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_object import Object
from litemind.media.types.media_text import Text


class ObjectConverter(BaseConverter):
    """Converts Object (Pydantic BaseModel) media to Text via JSON Markdown."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(Object, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        return media is not None and isinstance(media, Object)

    def convert(self, media: Any) -> List[MediaBase]:
        if not isinstance(media, Object):
            raise ValueError(f"Expected Object media, got {type(media)}")

        # Use the Object's built-in method to convert to Text
        return [media.to_markdown_text_media()]
