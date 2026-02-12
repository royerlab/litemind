"""Abstract base class for media converters in the conversion pipeline.

Defines the :class:`BaseConverter` interface that all concrete converters
must implement, including ``rule()``, ``can_convert()``, and ``convert()``.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Type

from litemind.media.media_base import MediaBase


class BaseConverter(ABC):
    """Abstract base class for media converters.

    A converter transforms one media type into one or more simpler or more
    normalised media types. For example, a video converter might produce
    image frames and a text transcription.
    """

    @abstractmethod
    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare which media type conversions this converter supports.

        Each tuple maps a source media type to the list of media types that
        the conversion may produce.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A list of ``(source_type, [target_type, ...])`` tuples.
        """

    @abstractmethod
    def can_convert(self, media: MediaBase) -> bool:
        """Check whether this converter can handle the given media instance.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if this converter can convert the media.
        """

    @abstractmethod
    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert the given media into one or more target media.

        Parameters
        ----------
        media : MediaBase
            The media to convert.

        Returns
        -------
        List[MediaBase]
            The resulting media objects. A single source media may produce
            multiple outputs (e.g. a video yields frames and audio).
        """
