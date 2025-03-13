from abc import ABC, abstractmethod
from typing import List, Optional

from litemind.agent.augmentations.hierarchical_document import HierarchicalDocument


class TextSplitter(ABC):
    """
    Abstract base class for text splitters.
    
    Text splitters are responsible for breaking documents into smaller chunks
    that can be processed more efficiently by embedding models and LLMs.
    """
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Parameters
        ----------
        text: str
            The text to split.
            
        Returns
        -------
        List[str]
            The text chunks.
        """
        pass
    
    def create_documents(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[HierarchicalDocument]:
        """
        Create HierarchicalDocument objects from text chunks.
        
        Parameters
        ----------
        texts: List[str]
            The text chunks.
        metadatas: Optional[List[dict]]
            Metadata for each text chunk.
            
        Returns
        -------
        List[HierarchicalDocument]
            The created documents.
        """
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        return [HierarchicalDocument(content=text, metadata=metadata) 
                for text, metadata in zip(texts, metadatas)]
    
    def split_documents(self, documents: List[HierarchicalDocument]) -> List[HierarchicalDocument]:
        """
        Split HierarchicalDocument objects into smaller chunks.
        
        Parameters
        ----------
        documents: List[HierarchicalDocument]
            The documents to split.
            
        Returns
        -------
        List[HierarchicalDocument]
            The split documents.
        """
        result = []
        for doc in documents:
            chunks = self.split_text(doc.content)
            for chunk in chunks:
                # Create a child document with the chunk content
                child_doc = HierarchicalDocument(
                    content=chunk,
                    metadata=doc.metadata,
                )
                # Add the child to the parent
                doc.add_child(child_doc)
                result.append(child_doc)
                
        return result









