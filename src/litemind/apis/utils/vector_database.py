import os
from typing import Dict, Optional

import chromadb
from chrombadb.utils.embedding_function import (
    GoogleGenerativeAiEmbeddingFunction,
    OllamaEmbeddingFunction,
    OpenAIEmbeddingFunction,
    VoyageAIEmbeddingFunction,
)

from litemind.utils.database import is_chromadb_available


class ChromaDB:
    """
    This class is a basic implementation of a
    Chroma database that can be saved locally
    """

    def __init__(
        self,
        persistent_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> None:
        # verify that Chromadb is installed
        if not is_chromadb_available():
            raise ImportError("chromadb is not available. Please install it to use it.")
        # initialize a chroma client that will be saved locally
        if persistent_path is None:
            current_path = os.getcwd()
            self.chroma_client = chromadb.PersistentClient(
                path=os.path.join(current_path, "vector_storage")
            )
        # TODO check about the the abosulte path
        # check the collection name
        if collection_name is None:
            self.collection_name = "VectorStorage"
        # check the embedding method
        if embedding_model is None:
            self.embedding_model = "text-embedding-3-small"

        # create client with given variables
        self.chroma_client = chromadb.PersistentClient(path=persistent_path)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.metadata = metadata

        # select embedding method for the database

        # select the correct embedding function
        if self.embedding_model == "openai":
            api_key = os.getenv(
                "OPENAI_API_KEY"
            )  # to change or see how to incorporate with the basi_API
            self.embedding_method = OpenAIEmbeddingFunction(
                api_key=api_key, model_name=self.embedding_model
            )
        elif self.embedding_model == "google":
            api_key = os.getenv("GOOGLE_API_KEY")
            self.embedding_method = GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
        elif self.embedding_model == "olloma":
            url = "url"
            self.embedding_method = OllamaEmbeddingFunction(
                url=url, model_name=self.embedding_model
            )
        else:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            self.embedding_method = VoyageAIEmbeddingFunction(
                api_key=api_key, model_name=self.embedding_model
            )

    def create_collection(self) -> chromadb.Collection:
        """
        This function create a chromadb collection

        Return
        -----------------
        The collection database created
        """

        # create the collection
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_method,
            metadata=self.metadata,
        )

        return self.collection

    def get_collection(self, name: str):
        """
        This function return an existing collection
        """

        self.collection = self.chroma_client.get_collection(name=name)

        return self.collection

    def delete_collection(self, name: str) -> None:

        self.chroma_client.delete_collection(name=name)

        print(f"The collection {name} was deleted.")

    def rename_collection(self, name: str) -> None:

        self.chroma_client.modify(name=name)
