from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_document import Document
from litemind.media.types.media_text import Text
from litemind.utils.file_types.file_types import classify_uri
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class DocumentConverterTxt(BaseConverter):
    """
    Converter for text documents media types.
    Converts text documents media to Text media.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(Document, [Text])]

    def can_convert(self, media: MediaBase) -> bool:

        # Check is the media is None:
        if media is None:
            return False

        # Check if the media is a Document:
        if not isinstance(media, Document):
            return False

        # Check if the file is a PDF:
        file_type = classify_uri(media.uri)
        if file_type not in {"text", "code", "script"}:
            return False

        # By default, we can convert the media, unless the previous checks failed:
        return True

    def convert(self, media: MediaBase) -> List[MediaBase]:

        if not isinstance(media, Document):
            raise ValueError(f"Expected Document media, got {type(media)}")

        # Normalise uri to local path:
        document_path = uri_to_local_file_path(media.uri)

        # get filename:
        filename = media.get_filename()

        # Get the file extension:
        extension = media.get_extension()

        # Read the content of the text document from file:
        with open(document_path, "r", encoding="utf-8") as file:
            text_content = file.read()

        # Wrap file in markdown quotes and give filename:
        text_content = f"{filename}\n```{extension}\n{text_content}\n```"

        # Create a Text media object:
        text_media = Text(text_content)

        # Return the converted media:
        return [text_media]
