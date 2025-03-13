import uuid
from typing import Optional, List

from pydantic import BaseModel


class Document:
    """
    A document that can be stored in a vector database and retrieved for augmentation.

    """
    content: str
    summary: Optional[str] = None
    metadata: Optional[dict] = None
    id: Optional[str] = None
    embedding: Optional[List[float]] = None
    score: Optional[float] = None

    def __init__(self,
                 content: str,
                 summary: Optional[str] = None,
                 metadata: Optional[dict] = None,
                 id: Optional[str] = None,
                 score: Optional[float] = None):
        """
        Create a new document.

        Parameters
        ----------
        content: str
            The text content of the document.
        summary: Optional[str]
            A summary of the document content.
        metadata: Optional[dict]
            Additional metadata about the document.
        id: Optional[str]
            A unique identifier for the document.
        score: Optional[float]
            A relevance score, usually assigned during retrieval.
        """
        self.content = content
        self.summary = summary
        self.metadata = metadata
        self.id = id or str(uuid.uuid4())
        self.score = score

    def __copy__(self):
        return Document(
            content=self.content,
            summary=self.summary,
            metadata=self.metadata,
            id=self.id,
            score=self.score,
        )

    def __contains__(self, item):
        return item in self.content

    def __str__(self) -> str:
        return self.content

    def __hash__(self) -> int:
        return hash(self.content)

    def __eq__(self, other) -> bool:
        return self.content == other.content

    def __ne__(self, other) -> bool:
        return not self == other


