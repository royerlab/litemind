from typing import Iterator, Optional, Union

from pydantic import BaseModel

from litemind.agent.augmentations.augmentation_base import AugmentationBase
from litemind.agent.augmentations.information.information import Information
from litemind.media.media_base import MediaBase
from litemind.media.types.media_object import Object
from litemind.media.types.media_text import Text


class AugmentationDefault(AugmentationBase):
    """
    Default implementation of the AugmentationBase class.
    This class is meant to be subclassed by concrete augmentation implementations.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        """Initialize the augmentation with a name and optional description."""
        super().__init__(name, description)

    @staticmethod
    def _normalize_query(
        query: Union[str, BaseModel, MediaBase, Information],
    ) -> MediaBase:
        """
        Normalize a query to a string regardless of its type.

        Parameters
        ----------
        query: Union[str, BaseModel, MediaBase, Information]
            The query to normalize.

        Returns
        -------
        MediaBase
            The query normalized to a media object.
        """

        if isinstance(query, str):
            # For string queries, we wrap as a text media:
            return Text(query)
        elif isinstance(query, BaseModel):
            # For BaseModel objects, we wrap into an Object media:
            return Object(query)
        elif isinstance(query, MediaBase):
            # For MediaBase objects, we just keep as is as it is supported by Vector Databases.
            return query
        elif isinstance(query, Information):
            # Get the media in the information:
            media = query.media
            return media
        else:
            raise ValueError(
                f"Unsupported query type: {type(query)}. Supported types are: str, BaseModel, MediaBase, Information."
            )

    def get_relevant_informations_iterator(
        self,
        query: Union[str, BaseModel, MediaBase, Information],
        k: int = 5,
        threshold: float = 0.0,
    ) -> Iterator[Information]:
        """
        Retrieve informations as an iterator, normalizing the query first.

        Parameters
        ----------
        query: Union[str, BaseModel]
            The query to retrieve informations for.
        k: int
            The maximum number of informations to retrieve.
        threshold: float
            The score threshold_dict for the augmentation. If the score is below this value, the augmentation will not be used.

        Returns
        -------
        Iterator[Information]
            An iterator over relevant informations.
        """

        # Normalize the query to a string:
        normalized_query: MediaBase = self._normalize_query(query)

        # Use the base class method to get relevant informations:
        for info in self.get_relevant_informations(
            normalized_query, k=k, threshold=threshold
        ):
            yield info
