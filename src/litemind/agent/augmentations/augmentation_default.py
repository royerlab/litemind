"""Default partial implementation of the augmentation interface.

This module provides ``AugmentationDefault``, a convenience base class that
sits between ``AugmentationBase`` and concrete augmentation implementations.
It normalizes heterogeneous query types (strings, Pydantic models, media
objects, and ``Information`` instances) into ``MediaBase`` objects and
supplies a default iterator-based retrieval method.
"""

from typing import Iterator, Optional, Union

from pydantic import BaseModel

from litemind.agent.augmentations.augmentation_base import AugmentationBase
from litemind.agent.augmentations.information.information import Information
from litemind.media.media_base import MediaBase
from litemind.media.types.media_object import Object
from litemind.media.types.media_text import Text


class AugmentationDefault(AugmentationBase):
    """Default partial implementation of AugmentationBase with query normalization.

    Provides ``_normalize_query`` to convert various query types (str,
    BaseModel, MediaBase, Information) into MediaBase objects, and a
    default ``get_relevant_informations_iterator`` implementation.
    Subclasses must still implement ``get_relevant_informations``.
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

        if isinstance(query, Information):
            # Get the media in the information:
            return query.media
        elif isinstance(query, str):
            # For string queries, we wrap as a text media:
            return Text(query)
        elif isinstance(query, MediaBase):
            # For MediaBase objects, we just keep as is as it is supported by Vector Databases.
            return query
        elif isinstance(query, BaseModel):
            # For BaseModel objects, we wrap into an Object media:
            return Object(query)
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
        query : Union[str, BaseModel, MediaBase, Information]
            The query to retrieve informations for.
        k : int
            The maximum number of informations to retrieve.
        threshold : float
            Minimum relevance score. Results below this threshold are
            excluded.

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
