from abc import abstractmethod, ABC
from typing import List

from litemind.media.media_base import MediaBase


class BaseConverter(ABC):

    """
    Base class for all converters.
    """

    @abstractmethod
    def can_convert(self, media: MediaBase) -> bool:
        """
        Returns a True if the the converter can convert the given media.

        PARAMETERS
        ----------
        media : MediaBase
            The media to convert.

        RETURNS
        -------
        bool
            True if the converter can convert the given media, False otherwise.
        """

    @abstractmethod
    def convert(self, media: MediaBase) -> List[MediaBase]:
        """
        Converts the given media to a different media. Typically this will be a 'simpler' and more 'normalised' media type.

        PARAMETERS
        ----------
        media : MediaBase
            The media to convert.

        RETURNS
        -------
        List[MediaBase]
            The converted medias. A single media might convert to a list of medias, for example a video can be converted to individual image frames and the audio transciption.
        """

