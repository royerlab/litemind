from typing import Callable, List, Optional, Sequence


from litemind.agent.augmentations.augmentation_base import AugmentationBase
from litemind.apis.base_api import BaseApi
from litemind.rag.vector_db.base_vector_db import BaseVectorDatabase


class DefaultVectorDatabase(AugmentationBase, BaseVectorDatabase):
    """
    Abstract class for vector databases that use a default embedding function.
    """
    
    def __init__(
        self,
        name: str = "QdrantVectorDB",
        description: Optional[str] = None,
        embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
        embedding_length: int = 768,
        api: BaseApi = None,
    ):
        """
        Initialize the Qdrant vector database.
        
        Parameters
        ----------
        name: str
            The name of the augmentation.
        description: Optional[str]
            A description of the augmentation.
        embedding_function: Optional[Callable]
            A function that takes a string and returns an embedding vector.
        embedding_length: int
            The length of the embedding vectors.
        api: BaseApi
            The API to use for embedding. If not provided, uses litemind's default API.

        """
        super().__init__(name=name, description=description)

        if api is None:
            from litemind import CombinedApi
            self.api = CombinedApi()
        else:
            self.api = api

        self.embedding_function = embedding_function or self._default_embedding_function
        self.embedding_length = embedding_length

    
    def _default_embedding_function(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        """
        Default embedding function using litemind's API.
        
        Parameters
        ----------
        texts: str
            List of texts (strings) to embed.
            
        Returns
        -------
        List[List[float]]
            The embedding vectors.
        """
        # Embed the texts:
        embeddings = self.api.embed_texts(texts=texts,
                                          dimensions=self.embedding_length)

        # Return the embeddings:
        return embeddings

