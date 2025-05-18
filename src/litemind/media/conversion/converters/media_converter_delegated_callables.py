from typing import List, Tuple, Type

from litemind.apis.base_api import BaseApi
from litemind.apis.model_features import ModelFeatures
from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video


class MediaConverterApi(BaseConverter):
    """
    Converter for Image, Audio, Video and Document media relying on a Litemind provider API.

    """

    def __init__(self, api: BaseApi):
        """
        Initialize the converter with a Litemind provider API.
        These functions should take a string (the media URI) and return a string (the description).
        The functions should be able to handle the media URI and return a description.

        Parameters
        ----------
        api: BaseApi
            The Litemind provider API to use for conversion.
        """

        super().__init__()
        self.api: BaseApi = api

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        # Define the conversion rules for each media type
        return [(Image, [Text]), (Audio, [Text]), (Video, [Text]), (Document, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        # Check if the media is one of the supported types
        return media is not None and isinstance(media, (Image, Audio, Video, Document))

    def convert(self, media: MediaBase) -> List[MediaBase]:

        try:
            if isinstance(media, Image) and self.api.has_model_support_for(
                ModelFeatures.Image
            ):
                description = self.api.describe_image(media.uri)
            elif isinstance(media, Audio) and self.api.has_model_support_for(
                ModelFeatures.Audio
            ):
                description = self.api.describe_audio(media.uri)
            elif isinstance(media, Video) and self.api.has_model_support_for(
                ModelFeatures.Video
            ):
                description = self.api.describe_video(media.uri)
            elif isinstance(media, Document) and self.api.has_model_support_for(
                ModelFeatures.Document
            ):
                description = self.api.describe_document(media.uri)
            else:
                raise ValueError(
                    f"Expected Image, Audio, Video, or Document media, got {type(media)}"
                )

            # Create a Text media with the description
            text_media = Text(description)

            return [text_media]

        except Exception as e:

            # Log the error
            import traceback

            traceback.print_exc()

            # If any error occurs, return the original media
            return [media]
