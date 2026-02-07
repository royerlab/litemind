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
    """Converts Image, Audio, Video, and Document media to Text via a LLM provider API.

    Delegates to the provider's ``describe_*`` methods (e.g.
    ``describe_image``, ``describe_audio``) to generate text descriptions.
    Only supports media types for which the underlying model has the
    corresponding feature (Image, Audio, Video, Document).
    """

    def __init__(self, api: BaseApi):
        """Initialise the converter with a litemind provider API.

        Parameters
        ----------
        api : BaseApi
            The provider API used to generate text descriptions of media.
        """

        super().__init__()
        self.api: BaseApi = api

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare all theoretically possible API-based media conversions.

        Returns a static list of all media-to-text conversions that the
        API *could* perform. The actual feature check is deferred to
        ``can_convert()`` which is called at conversion time, not during
        graph construction. This avoids a circular dependency where
        ``rule()`` calls ``has_model_support_for()`` which calls
        ``can_convert_within()`` which calls ``rule()`` again.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            Rules for each media type the API could potentially describe.
        """
        return [
            (Image, [Text]),
            (Audio, [Text]),
            (Video, [Text]),
            (Document, [Text]),
        ]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the API can describe the given media type.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media type is supported by the underlying model.
        """
        if isinstance(media, Image) and self.api.has_model_support_for(
            ModelFeatures.Image
        ):
            return True
        elif isinstance(media, Audio) and self.api.has_model_support_for(
            ModelFeatures.Audio
        ):
            return True
        elif isinstance(media, Video) and self.api.has_model_support_for(
            ModelFeatures.Video
        ):
            return True
        elif isinstance(media, Document) and self.api.has_model_support_for(
            ModelFeatures.Document
        ):
            return True
        else:
            return False

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Generate a text description of the media using the LLM API.

        Falls back to returning the original media if the API call fails.

        Parameters
        ----------
        media : MediaBase
            The media to describe.

        Returns
        -------
        List[MediaBase]
            A single-element list containing a Text description, or the
            original media if an error occurs.
        """

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

        except Exception:

            # Log the error
            import traceback

            traceback.print_exc()

            # If any error occurs, return the original media
            return [media]
