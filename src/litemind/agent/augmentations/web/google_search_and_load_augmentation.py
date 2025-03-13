from typing import Dict, List, Optional, Union

from arbol import aprint, asection

from litemind.agent.augmentations.augmentation_base import AugmentationBase, Document
from litemind.agent.augmentations.web.google_search_augmentation import GoogleSearchAugmentation
from litemind.agent.augmentations.utils.web_page_loader import WebPageLoader


class GoogleSearchAndLoadAugmentation(AugmentationBase):
    """
    An augmentation that combines Google Search and web page loading.

    This augmentation first performs a Google search, then loads and indexes
    the top search results for retrieval.
    """

    def __init__(
            self,
            vector_database,
            api_key: Optional[str] = None,
            cse_id: Optional[str] = None,
            name: str = "GoogleSearchAndLoad",
            description: Optional[str] = None,
            cache_results: bool = True,
    ):
        """
        Initialize the Google Search and Load augmentation.

        Parameters
        ----------
        vector_database
            A vector database to store the web page documents.
        api_key: Optional[str]
            The Google API key. If not provided, will look for GOOGLE_API_KEY environment variable.
        cse_id: Optional[str]
            The Custom Search Engine ID. If not provided, will look for GOOGLE_CSE_ID environment variable.
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        cache_results: bool
            Whether to cache search results to avoid redundant API calls.
        """
        super().__init__(name=name, description=description or "Google Search with web page loading")
        self.google_search = GoogleSearchAugmentation(api_key=api_key, cse_id=cse_id)
        self.vector_database = vector_database
        self.loaded_urls = set()
        self.cache_results = cache_results
        self.query_cache: Dict[str, List[Document]] = {}

    def get_relevant_documents(self, query: Union[str, Document], k: int = 5) -> List[Document]:
        """
        Get relevant documents by searching Google and loading web pages.

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
        # Extract query text
        if isinstance(query, Document):
            query_text = query.content
        else:
            query_text = query

        # Check cache if enabled
        if self.cache_results and query_text in self.query_cache:
            return self.query_cache[query_text][:k]

        with asection("Performing Google search and loading web pages"):
            # First, get search results
            search_results = self.google_search.get_relevant_documents(query_text, k=k)

            # Load and index the web pages
            for result in search_results:
                url = result.metadata.get("url")
                if url and url not in self.loaded_urls:
                    try:
                        document = WebPageLoader.load_url(url)
                        self.vector_database.add_documents([document])
                        self.loaded_urls.add(url)
                    except Exception as e:
                        aprint(f"Error loading URL {url}: {e}")

            # Query the vector database
            documents = self.vector_database.get_relevant_documents(query_text, k=k)

            # Cache the results if enabled
            if self.cache_results:
                self.query_cache[query_text] = documents

            return documents
