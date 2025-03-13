from typing import Callable, Dict, List, Optional, Any

import numpy as np

from litemind.agent.augmentations.document import Document


class HierarchicalDocument(Document):
    """
    A document that can have a hierarchical structure with parent-child relationships.
    
    This is useful for representing complex documents like PDFs with chapters,
    code repositories with files and folders, or any other nested document structure.
    
    Attributes
    ----------
    parent_id: Optional[str]
        The ID of the parent document.
    children: List[HierarchicalDocument]
        Child documents.
    """

    parent_id: Optional[str] = None
    children: List["HierarchicalDocument"]

    def add_child(self, document: "HierarchicalDocument") -> None:
        """
        Add a child document to this document.
        
        Parameters
        ----------
        document: HierarchicalDocument
            The child document to add.
        """
        document.parent_id = self.id
        self.children.append(document)
    
    def compute_embedding(self, embedding_function: Optional[Callable[[str], List[float]]] = None) -> List[float]:
        """
        Compute an embedding for this document.
        
        If the document has children, compute embeddings for them as well and average.
        
        Parameters
        ----------
        embedding_function: Optional[Callable[[str], List[float]]]
            A function that takes a string and returns an embedding vector.
            If not provided, uses litemind's default embedding function.
            
        Returns
        -------
        List[float]
            The computed embedding.
        """
        if embedding_function is None:
            from litemind import CombinedApi
            api = CombinedApi()
            embedding_function = lambda text: api.embed_texts([text])[0]
        
        # If we already have an embedding, just return it
        if self.embedding is not None:
            return self.embedding
        
        # If we don't have children, compute the embedding for our content
        if not self.children:
            self.embedding = embedding_function(self.content)
            return self.embedding
        
        # If we have children, make sure they all have embeddings
        for child in self.children:
            if child.embedding is None:
                child.compute_embedding(embedding_function)
            
        # Average the embeddings of our children
        embeddings = [child.embedding for child in self.children] + [embedding_function(self.content)]
        self.embedding = np.mean(embeddings, axis=0).tolist()

        # Return the computed embedding
        return self.embedding

    def get_all_documents(self) -> List[Document]:
        """
        Get all documents in the hierarchy as flat documents.
        
        Returns
        -------
        List[Document]
            A list of all documents in the hierarchy.
        """
        docs = [self.to_document()]
        for child in self.children:
            docs.extend(child.get_all_documents())
        return docs
    
    def summarize(self, summarization_function: Optional[Callable[[str], str]] = None) -> str:
        """
        Generate a summary for this document.
        
        Parameters
        ----------
        summarization_function: Optional[Callable[[str], str]]
            A function that takes a document text and returns a summary.
            If not provided, uses litemind to summarize.
            
        Returns
        -------
        str
            The generated summary.
        """
        if summarization_function is None:
            # Use litemind to summarize
            from litemind import CombinedApi
            from litemind.agent.messages.message import Message
            
            api = CombinedApi()
            messages = [
                Message(role="system", text="You are a helpful assistant that summarizes documents."),
                Message(role="user", text=f"Please summarize the following document in a concise way:\n\n{self.content}")
            ]
            response = api.generate_text(messages=messages)
            self.summary = response[0].to_plain_text()
        else:
            self.summary = summarization_function(self.content)
            
        return self.summary


def split_document_by_character(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[HierarchicalDocument]:
    """
    Split a document into chunks based on character count.
    
    Parameters
    ----------
    text: str
        The text to split.
    chunk_size: int
        The target size of each chunk.
    chunk_overlap: int
        The overlap between consecutive chunks.
        
    Returns
    -------
    List[HierarchicalDocument]
        A list of hierarchical documents.
    """
    if len(text) <= chunk_size:
        return [HierarchicalDocument(content=text)]
    
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        
        # Try to find a good split point (newline or space)
        if end < len(text):
            # Look for a newline first
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start:
                end = newline_pos + 1
            else:
                # No newline, look for a space
                space_pos = text.rfind(' ', start, end)
                if space_pos > start:
                    end = space_pos + 1
        
        chunks.append(HierarchicalDocument(content=text[start:end]))
        start = end - chunk_overlap
    
    return chunks


def create_document_hierarchy(
    text: str,
    max_chunk_size: int = 1000,
    chunk_overlap: int = 200,
    summarization_function: Optional[Callable[[str], str]] = None
) -> HierarchicalDocument:
    """
    Create a document hierarchy from a text.
    
    Parameters
    ----------
    text: str
        The text to process.
    max_chunk_size: int
        Maximum size of each chunk.
    chunk_overlap: int
        The overlap between consecutive chunks.
    summarization_function: Optional[Callable[[str], str]]
        Optional function to generate summaries.
        
    Returns
    -------
    HierarchicalDocument
        The root of the document hierarchy.
    """
    # Create a root document
    root = HierarchicalDocument(content=text)
    
    # If the text is small enough, just return the root
    if len(text) <= max_chunk_size:
        return root
    
    # Split the document into chunks
    chunks = split_document_by_character(text, max_chunk_size, chunk_overlap)
    for chunk in chunks:
        root.add_child(chunk)
    
    # Generate a summary for the root
    if summarization_function:
        root.summarize(summarization_function)
    else:
        # Only summarize if the text is longer than the chunk size
        if len(text) > max_chunk_size:
            root.summarize()
    
    return root


def create_documents_from_texts(texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
    """
    Create documents from a list of texts.
    
    Parameters
    ----------
    texts: List[str]
        The texts to convert to documents.
    metadata: Optional[List[Dict[str, Any]]]
        Optional metadata for each document.
        
    Returns
    -------
    List[Document]
        The created documents.
    """
    documents = []
    
    if metadata is None:
        metadata = [{} for _ in texts]
    
    for text, meta in zip(texts, metadata):
        documents.append(Document(content=text, metadata=meta))
    
    return documents
