from typing import List, Optional, Union

from arbol import aprint

from litemind.agent.augmentations.augmentation_base import AugmentationBase, Document
from litemind.agent.augmentations.utils.web_page_loader import WebPageLoader


class WebPageAugmentation(AugmentationBase):
    """
    An augmentation that retrieves and loads web pages.
    
    This augmentation works with a list of URLs that are loaded and made available
    for retrieval.
    """
    
    def __init__(
        self,
        vector_database,
        urls: Optional[List[str]] = None,
        name: str = "WebPage",
        description: Optional[str] = None,
    ):
        """
        Initialize the web page augmentation.
        
        Parameters
        ----------
        vector_database
            A vector database to store the web page documents.
        urls: Optional[List[str]]
            A list of URLs to pre-load.
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        """
        super().__init__(name=name, description=description or "Web page documents")
        self.vector_database = vector_database
        self.loaded_urls = set()
        
        # Load any provided URLs
        if urls:
            for url in urls:
                self.add_url(url)
    
    def add_url(self, url: str, include_headers: bool = True) -> bool:
        """
        Add a web page to the augmentation.
        
        Parameters
        ----------
        url: str
            The URL to add.
        include_headers: bool
            Whether to include headers in the document content.
            
        Returns
        -------
        bool
            True if the URL was successfully added, False otherwise.
        """
        # Skip if already loaded
        if url in self.loaded_urls:
            return True
            
        try:
            # Load the web page
            document = WebPageLoader.load_url(url, include_headers=include_headers)
            
            # Add the document to the vector database
            self.vector_database.add_documents([document])
            
            # Mark the URL as loaded
            self.loaded_urls.add(url)
            
            return True
        except Exception as e:
            aprint(f"Error adding URL {url}: {e}")
            return False
    
    def get_relevant_documents(self, query: Union[str, Document], k: int = 5) -> List[Document]:
        """
        Get relevant documents from the loaded web pages.
        
        Parameters
        ----------
        query: Union[str, Document]
            The query to search for.
        k: int
            The number of documents to return.
            
        Returns
        -------
        List[Document]
            A list of relevant documents.
        """
        # Delegate to the vector database
        return self.vector_database.get_relevant_documents(query, k=k)


