from typing import Dict, List, Optional, Union

from arbol import aprint

from litemind.agent.augmentations.augmentation_base import AugmentationBase, Document


class AugmentationSet:
    """
    A set of augmentations that can be used by an agent.
    
    This is similar to the ToolSet class but for augmentations.
    """
    
    def __init__(self, augmentations: Optional[List[AugmentationBase]] = None):
        """
        Create a new augmentation set.
        
        Parameters
        ----------
        augmentations: Optional[List[AugmentationBase]]
            Initial list of augmentations.
        """
        self.augmentations: List[AugmentationBase] = augmentations or []
        
    def add_augmentation(self, augmentation: AugmentationBase):
        """
        Add an augmentation to the set.
        
        Parameters
        ----------
        augmentation: AugmentationBase
            The augmentation to add.
        """
        self.augmentations.append(augmentation)
        
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
                return True
        return False
    
    def search_all(self, query: Union[str, Document], k: int = 5) -> Dict[str, List[Document]]:
        """
        Search all augmentations and return the results.
        
        Parameters
        ----------
        query: Union[str, Document]
            The query to search for.
        k: int
            The number of results to return from each augmentation.
            
        Returns
        -------
        Dict[str, List[Document]]
            A dictionary mapping augmentation names to lists of relevant documents.
        """
        results = {}
        
        for augmentation in self.augmentations:
            try:
                docs = augmentation.get_relevant_documents(query, k=k)
                results[augmentation.name] = docs
            except Exception as e:
                aprint(f"Error searching augmentation {augmentation.name}: {e}")
                results[augmentation.name] = []
                
        return results
    
    def search_combined(self, query: Union[str, Document], k: int = 5) -> List[Document]:
        """
        Search all augmentations and combine the results.
        
        Parameters
        ----------
        query: Union[str, Document]
            The query to search for.
        k: int
            The total number of results to return.
            
        Returns
        -------
        List[Document]
            A combined list of relevant documents from all augmentations,
            sorted by relevance score.
        """
        all_docs = []
        
        for augmentation in self.augmentations:
            try:
                docs = augmentation.get_relevant_documents(query, k=k)
                # Add augmentation name to metadata
                for doc in docs:
                    if doc.metadata is None:
                        doc.metadata = {}
                    doc.metadata["augmentation"] = augmentation.name
                all_docs.extend(docs)
            except Exception as e:
                aprint(f"Error searching augmentation {augmentation.name}: {e}")
                
        # Sort by score (if available)
        all_docs.sort(key=lambda x: x.score if x.score is not None else 0, reverse=True)
        
        # Return top k results
        return all_docs[:k]
    
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
