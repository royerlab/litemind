from typing import Dict, List, Optional, Union

from arbol import aprint
from pydantic import BaseModel

from litemind.agent.augmentations.augmentation_base import AugmentationBase
from litemind.agent.augmentations.information.information import Information
from litemind.agent.augmentations.information.information_base import InformationBase
from litemind.media.media_base import MediaBase


class AugmentationSet:
    """A managed collection of augmentations for Retrieval-Augmented Generation.

    Each augmentation can have its own ``k`` (number of results) and
    ``threshold`` (minimum relevance score) settings. Provides methods
    to search all augmentations individually or combine results ranked
    by relevance score.
    """

    def __init__(self):
        """
        Create a new augmentation set.
        """

        # Set attributes:
        self.augmentations: List[AugmentationBase] = []
        self.k_dict: Dict[AugmentationBase, int] = {}
        self.threshold_dict: Dict[AugmentationBase, float] = {}

    def add_augmentation(
        self,
        augmentation: AugmentationBase,
        k: int = 5,
        threshold: float = 0.0,
    ):
        """
        Add an augmentation with per-augmentation retrieval settings.

        Parameters
        ----------
        augmentation : AugmentationBase
            The augmentation to add.
        k : int
            The maximum number of informations to retrieve from this
            augmentation.
        threshold : float
            Minimum relevance score. Results below this value are excluded.
        """
        self.augmentations.append(augmentation)
        self.k_dict[augmentation] = k
        self.threshold_dict[augmentation] = threshold

    def get_augmentation(self, name: str) -> Optional[AugmentationBase]:
        """
        Get an augmentation by name.

        Parameters
        ----------
        name: str
            The name of the augmentation.

        Returns
        -------
        Optional[AugmentationBase]
            The augmentation, or None if not found.
        """
        for augmentation in self.augmentations:
            if augmentation.name == name:
                return augmentation
        return None

    def list_augmentations(self) -> List[AugmentationBase]:
        """
        Get all augmentations in the set.

        Returns
        -------
        List[AugmentationBase]
            The list of augmentations.
        """
        return self.augmentations

    def augmentation_names(self) -> List[str]:
        """
        Get the names of all augmentations in the set.

        Returns
        -------
        List[str]
            The list of augmentation names.
        """
        return [aug.name for aug in self.augmentations]

    def remove_augmentation(self, name: str) -> bool:
        """
        Remove an augmentation by name.

        Parameters
        ----------
        name: str
            The name of the augmentation to remove.

        Returns
        -------
        bool
            True if the augmentation was removed, False if not found.
        """
        for i, aug in enumerate(self.augmentations):
            if aug.name == name:
                del self.augmentations[i]
                del self.k_dict[aug]
                del self.threshold_dict[aug]
                return True
        return False

    def search_all(
        self,
        query: Union[str, BaseModel, MediaBase, Information],
        k: Optional[int] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, List[InformationBase]]:
        """
        Search all augmentations and return results per augmentation.

        Parameters
        ----------
        query : Union[str, BaseModel, MediaBase, Information]
            The query to search for.
        k : Optional[int]
            Maximum results per augmentation. If None, uses each
            augmentation's configured ``k`` value.
        threshold : Optional[float]
            Minimum relevance score. If None, uses each augmentation's
            configured threshold.

        Returns
        -------
        Dict[str, List[InformationBase]]
            A dictionary mapping augmentation names to their result lists.
        """

        # Initialize results dictionary
        results = {}

        # Iterate over augmentations and get results
        for augmentation in self.augmentations:
            try:
                # Perform search:
                infos = self._search_raw(augmentation, k, query, threshold)

                # Add results to the results dictionary:
                results[augmentation.name] = infos

            except Exception as e:
                aprint(f"Error searching augmentation {augmentation.name}: {e}")
                results[augmentation.name] = []

        return results

    def search_combined(
        self, query: Union[str, InformationBase], k: int = 5, threshold: float = 0.0
    ) -> List[InformationBase]:
        """
        Search all augmentations and return a merged, ranked result list.

        Results from all augmentations are combined and sorted by
        descending relevance score. This assumes scores are comparable
        across augmentations.

        Parameters
        ----------
        query : Union[str, InformationBase]
            The query to search for.
        k : int
            The total number of top results to return across all
            augmentations.
        threshold : float
            Minimum relevance score. Results below this value are excluded.

        Returns
        -------
        List[InformationBase]
            The top ``k`` informations sorted by relevance score.
        """

        # Initialize an empty list to store all informations
        all_infos = []

        # Iterate over augmentations and get results
        for augmentation in self.augmentations:
            try:
                # Perform search:
                infos = self._search_raw(augmentation, k, query, threshold)

                # Add results to the all_infos list:
                all_infos.extend(infos)

            except Exception as e:
                aprint(f"Error searching augmentation {augmentation.name}: {e}")

        # Sort by score (if available)
        all_infos.sort(
            key=lambda x: x.score if x.score is not None else 0, reverse=True
        )

        # Return top k results
        return all_infos[:k]

    def _search_raw(self, augmentation, k, query, threshold):
        """
        Perform a search on a single augmentation and tag results.

        Retrieves the per-augmentation ``k`` and ``threshold`` if not
        overridden, calls ``get_relevant_informations``, and annotates
        each result's metadata with the augmentation name.

        Parameters
        ----------
        augmentation : AugmentationBase
            The augmentation to search.
        k : Optional[int]
            Maximum results. Falls back to the augmentation's configured
            value if None.
        query : Union[str, BaseModel, MediaBase, Information]
            The query to search for.
        threshold : Optional[float]
            Minimum relevance score. Falls back to the augmentation's
            configured value if None.

        Returns
        -------
        List[InformationBase]
            Relevant informations with augmentation name in metadata.
        """
        # Get the k value for this augmentation
        k_ = k if k is not None else self.k_dict.get(augmentation)

        # Get the threshold value for this augmentation
        threshold_ = (
            threshold
            if threshold is not None
            else self.threshold_dict.get(augmentation)
        )

        # Get the relevant informations from the augmentation
        infos = augmentation.get_relevant_informations(
            query, k=k_, threshold=threshold_
        )

        # Add augmentation name to metadata
        for info in infos:
            if info.metadata is None:
                info.metadata = {}
            info.metadata["augmentation"] = augmentation.name
        return infos

    def __getitem__(self, item) -> AugmentationBase:
        return self.augmentations[item]

    def __len__(self) -> int:
        return len(self.augmentations)

    def __iter__(self):
        return iter(self.augmentations)

    def __contains__(self, item: Union[AugmentationBase, str]) -> bool:
        if isinstance(item, str):
            return any(aug.name == item for aug in self.augmentations)
        return item in self.augmentations

    def __str__(self) -> str:
        return f"AugmentationSet({[aug.name for aug in self.augmentations]})"

    def __repr__(self) -> str:
        return self.__str__()
