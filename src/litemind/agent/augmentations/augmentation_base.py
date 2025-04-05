from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Union

from pydantic import BaseModel

from litemind.agent.augmentations.information.information import Information
from litemind.media.media_base import MediaBase


class AugmentationBase(ABC):
    """
    Abstract base class defining the interface for all augmentations.

    Augmentations provide additional context to the LLM by retrieving relevant
    informations based on a query.
    """

    def __init__(self, name: str, description: Optional[str] = None):
        """
        Create a new augmentation.

        Parameters
        ----------
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        """
        self.name = name
        self.description = description or f"{name} Augmentation"

    @abstractmethod
    def get_relevant_informations(
        self,
        query: Union[str, BaseModel, MediaBase, Information],
        k: int = 5,
        threshold: float = 0.0,
    ) -> List[MediaBase]:
        """
        Retrieve informations relevant to the query.

        Parameters
        ----------
        query: Union[str, BaseModel, MediaBase, Information]
            The query to retrieve informations for. Can be a string, a structured Pydantic model, a MediaBase object, or an Information object.
        k: int
            The maximum number of informations to retrieve.
        threshold: float
            The score threshold_dict for the augmentation. If the score is below this value, the augmentation will not be used.

        Returns
        -------
        List[InformationBase]
            A list of relevant informations.
        """
        pass

    def get_relevant_informations_iterator(
        self,
        query: Union[str, BaseModel, MediaBase, Information],
        k: int = 5,
        threshold: float = 0.0,
    ) -> Iterator[MediaBase]:
        """
        Retrieve informations relevant to the query as an iterator.
        This default implementation calls get_relevant_documents and yields from the result.

        Parameters
        ----------
        query: Union[str, BaseModel, MediaBase, Information]
            The query to retrieve informations for. Can be a string, a structured Pydantic model.
        k: int
            The maximum number of informations to retrieve.
        threshold: float
            The score threshold_dict for the augmentation. If the score is below this value, the augmentation will not be returned.

        Returns
        -------
        Iterator[InformationBase]
            An iterator over relevant informations.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __str__(self):
        return self.name
