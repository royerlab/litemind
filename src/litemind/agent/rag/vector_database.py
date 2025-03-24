import os
from typing import Dict, List, Optional

import chromadb
from litemind.agent.rag.chromadb_embeddings import EmbeddingModel
from litemind.agent.rag.text_splitting import (
    code_splitting,
    markdown_splitting,
    semantic_splitting,
)
from litemind.utils.database import is_chromadb_available


class ChromaDB:
    """
    This class is a basic implementation of a
    Chroma vectorestore client. It will save the
    the embedded vector into a local folder accessible by the user.
    """

    def __init__(
        self,
        persistent_path: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_function: Optional[EmbeddingModel] = None,
    ) -> None:
        """
        Initialize a Chroma vectorestore client.

        Parameters
        ----------
        persistent_path: Optional[str] = None
            The path where the vectorestore is saved. If
            not given, it will create a folder in the current
            path, called vector_store.
        collection_name: Optional[str] = None
            The name of the collection of the vectorestore. If it's not given
            it will be called VectoreStorage. It can be modify using the function rename_collection.
        embedding_function: Optional[EmbeddingModel] = None
            The function used to embedding the text. If not given it will raise an error. If the model selected
            or model name for embedding is not available, it will raise an error.

        Raises
        ------
        ImportError
            If the user doesn't have installed the library chromadb.
        TypeError
            If the user doesn't specify an embedding model.

        """
        # verify that Chromadb is installed
        if not is_chromadb_available():
            raise ImportError("chromadb is not available. Please install it to use it.")
        # initialize a chroma client that will be saved locally
        if persistent_path is None:
            current_path = os.getcwd()
            self.chroma_client = chromadb.PersistentClient(
                path=os.path.join(current_path, "vector_storage")
            )
            self.persistent_path = os.path.join(current_path, "vector_storage")
        else:
            self.chroma_client = chromadb.PersistentClient(path=persistent_path)
        # check the collection name
        if collection_name is None:
            self.collection_name = "VectorStorage"

        if embedding_function is None:
            raise TypeError("You need to define the embedding model to use.")

        # create client with given variables
        # save the settings
        self.settings = self.chroma_client.get_settings()
        # save the current path
        self.persistent_path = self.settings.persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.metadata: Optional[Dict] = None
        self.collection_client: Optional[chromadb.Collection] = None

    def create_collection(self) -> None:
        """
        This function create a chromadb collection

        Parameters
        ----------
        model_name: str
            The name of the model to use

        Raises
        ------
        RunTimeError
            If the creation of the collection fails.
        """
        try:
            # create the collection
            self.collection_client = self.chroma_client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata=self.metadata,
            )

        except Exception as e:
            raise RuntimeError(f"{e}")
        print(f"A collection was created with the name {self.collection_name}")

    def get_collection_name(self) -> str:
        """
        This function return the name of the collection.
        """
        return self.collection_name

    def get_collection(self, name: str) -> chromadb.Collection:
        """
        This function return an existing collection

        Parameters
        ----------
        name: str
            The name of the collection.

        Returns
        -------
            The Collection present at the persistent path

        Raises
        ------
        RunTimeError
            If the name of the collection is invalid or
            the collection has not being created yet.
        """
        try:
            collection = self.chroma_client.get_collection(name=name)
        except Exception as e:
            raise RuntimeError(f" {e}")

        return collection

    def delete_collection(self, name: str) -> None:
        """
        This function delete permanently a collection

        Parameters
        ----------
        name: str
            The name of the collection.

        Raises
        ------
        RunTimeError
            If the deletion of the collection fails.
        """
        try:
            self.chroma_client.delete_collection(name=name)
        except Exception as e:
            raise RuntimeError(f"Runtimerror {e}")

        print(f"The collection {self.collection_name} was deleted.")

    def rename_collection(self, name: str) -> None:
        """
        This function rename an existant collection

        Parameters
        ----------
        name: str
            The new name of the collection.

        Returns
        -------
            The Collection present at the persistent path

        Raises
        ------
        RunTimeError
            If the rename of the collection fails.
        """
        try:
            self.chroma_client.modify(name=name)
            self.collection_name = name
        except Exception as e:
            raise RuntimeError(f"Runtimerror {e}")

        print(f"The collection {self.collection_name} was renamed in {name}")

    def get_settings(self) -> chromadb.config.Settings:
        """
        This function return the settings of the client
        """
        return self.settings

    def add_vectors(self, document: List[dict], embedding_func: EmbeddingModel) -> None:
        """
        This function add a vector to a database.

        Parameters
        ----------
        document: List[dict]
            A List of dictonary of documents['id':id, 'metadata':metadata|None, 'text':chunk text] obtained by the function load_documents
        embedding function: EmbeddingModel
            The name of the embedding method to use with the specific providers(OpenAI, GeminiAI, OllomaAI, AnthropicAI)

        Return
        ------
            It will insert the embedded vector into the current collection. To verify the added embedding, use chroma_client.show_vectorstore.

        Raises
        ------
        ValueError
            If the document parameter is not a List of dictonaries.
        Exception
            If the embedding function doesn't calculate the embedding vectors.
        ValueError
            If the collection name is wrong or the collection wasn't created yet.

        """
        if not isinstance(document, list) and all(
            isinstance(doc, dict) for doc in document
        ):
            raise ValueError(
                f"The function expect a List of dictonaries. A {document} was given."
            )
        # extract the chunked from the document list
        documents_chunked_text = [doc["text"] for doc in document]
        # extract the metadata from the document list
        documents_metadata = [doc["metadata"] for doc in document]  # ex. title
        # extract the ids from the document list
        documents_ids = [doc["id"] for doc in document]
        # calculate embeddings
        try:
            document_embedding_vectors = embedding_func(input=documents_chunked_text)
        except Exception as e:
            print(f'Embedding failed: {e}"')
        # verify that collection exists
        if self.collection_client is None:
            raise ValueError(
                "It doesn't exist a collection. Please define one before using this function."
            )
        # insert the embeddings into the vectorstore
        self.collection_client.upsert(
            ids=documents_ids,
            embeddings=document_embedding_vectors,
            metadatas=documents_metadata,
            documents=documents_chunked_text,
        )

    def create_vectorestore_from_documents(
        self,
        document_list: List[dict],
        embeddin_function: Optional[EmbeddingModel] = None,
        size=1000,
        overlap=0,
        trim=False,
    ) -> None:
        """
        This function create a local vectordatabase from a List of documents.
        The content of the documents will be chunked, transformed into vectors
        and saved into a vectorestore.

        Parameters
        ----------
        document_list: List[dict]
            A List of documents saved into a dictonary
                {'id': file name, 'metadata': the metadata of the file, 'text': extracted text of the file}
        embedding_function: Optional[EmbeddingModel] = None
            The embedding function to use with the text of the documents.
        size: int | 1000
            The size of the chunks that will be created from the text of each document.
        overlap: int | 0
            The overlap of the chunks that will be created from the text of each document.
        trim: bool
            If False the chunked text is not trim().

        Raises
        ------
        TyperError
            If the documents contains an unsupported document splitter.

        """

        documents = []
        for doc in document_list:
            # split the text of the documents into chunks
            if doc["metadata"]["filetype"] in ["txt", "docs", "docx"]:
                doc_chunks = semantic_splitting(
                    text=doc["text"], size=size, overlap=overlap, trim=trim
                )
            elif doc["metadata"]["filetype"] in ["md", "pdf"]:
                doc_chunks = markdown_splitting(
                    text=doc["text"], size=size, overlap=overlap, trim=trim
                )
            elif doc["metadata"]["filetype"] in [
                "py",
                "java",
                "html",
                "c",
                "cpp",
                "cs",
                "json",
                "sh",
            ]:
                doc_chunks = code_splitting(
                    language=doc["metadata"]["filetype"],
                    text=doc["text"],
                    size=size,
                    overlap=overlap,
                    trim=trim,
                )
            else:
                raise TypeError("The document contains an unsupported filetype.")

            # add the different chunks into lists of dict
            for i, chunk in enumerate(doc_chunks):
                doc["metadata"]["chunk"] = f"chunk_{i}"
                documents.append(
                    {
                        "id": f'{doc["id"]}_chunk{i}',
                        "metadata": doc["metadata"],
                        "text": chunk,
                    }
                )

        # add vectors to the database
        self.add_vectors(document=documents, embedding_func=embeddin_function)

    def show_vectorstore(self, show_lines: Optional[int]) -> None:
        """
        This function evoke a method to print the data contained into the
        vectorstore

        Parameters
        ----------
        show_line: int
            The number of line to show. By default is 20.

        Returns
        -------
            It print the vectore store containing
                ids: the ids of the text contained
                embeddings: the array of float of the text
                metadatas: A list of dictonary of metadata of the documents
                documents: The raw text saved
        """

        if show_lines is None:
            show_lines = 20

        print(self.collection_client.peek(show_lines))
