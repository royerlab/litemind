import os
from typing import Dict, List, Optional, Union

import requests
from arbol import aprint, asection
from bs4 import BeautifulSoup

from litemind.agent.augmentations.augmentation_base import AugmentationBase, Document


class GoogleSearchAugmentation(AugmentationBase):
    """
    An augmentation that retrieves information from Google Search.
    
    This augmentation requires the Google Search API to be set up and
    the API key and Custom Search Engine ID to be provided.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        cse_id: Optional[str] = None,
        name: str = "GoogleSearch",
        description: Optional[str] = None,
    ):
        """
        Initialize the Google Search augmentation.
        
        Parameters
        ----------
        api_key: Optional[str]
            The Google API key. If not provided, will look for GOOGLE_API_KEY environment variable.
        cse_id: Optional[str]
            The Custom Search Engine ID. If not provided, will look for GOOGLE_CSE_ID environment variable.
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        """
        super().__init__(name=name, description=description or "Google Search results")
        self.api_key = api_key or os.environ.get("GOOGLE_SEARCH_API_KEY")
        self.cse_id = cse_id or os.environ.get("GOOGLE_CSE_ID")
        
        if not self.api_key:
            raise ValueError("Google API key not found. Provide it or set GOOGLE_SEARCH_API_KEY environment variable.")
       # if not self.cse_id:
       #     raise ValueError("Custom Search Engine ID not found. Provide it or set GOOGLE_CSE_ID environment variable.")
    
    def get_relevant_documents(self, query: Union[str, Document], k: int = 5) -> List[Document]:
        """
        Get relevant documents from Google Search.
        
        Parameters
        ----------
        query: Union[str, Document]
            The query to search for.
        k: int
            The number of search results to return.
            
        Returns
        -------
        List[Document]
            A list of documents representing search results.
        """
        # Extract query text
        if isinstance(query, Document):
            query_text = query.content
        else:
            query_text = query
        
        # Construct the API URL
        url = "https://www.googleapis.com/customsearch/v1"

        # Set the query parameters
        params = {
            "key": self.api_key,
            "q": query_text,
            "num": min(k, 10),  # Google API limit is 10
        }

        # Add the Custom Search Engine ID if available
        if self.cse_id:
            params["cx"] = self.cse_id

        
        with asection(f"Performing Google search for: '{query_text}'"):
            # Make the API request
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                aprint(f"Error in Google Search API: {response.status_code}")
                aprint(response.text)
                return []
            
            # Parse the response
            search_results = response.json()
            
            # Extract the search results
            documents = []
            if "items" in search_results:
                for item in search_results["items"]:
                    # Create a document for each search result
                    content = f"Title: {item.get('title', '')}\n"
                    content += f"URL: {item.get('link', '')}\n"
                    content += f"Snippet: {item.get('snippet', '')}\n"
                    
                    metadata = {
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "source": "google_search",
                    }
                    
                    documents.append(Document(content=content, metadata=metadata))
                    
                    aprint(f"Found result: {item.get('title', '')}")
            
            return documents

