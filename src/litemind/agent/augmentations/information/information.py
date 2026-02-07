import uuid
from typing import Any, Callable, List, Optional, Type

from numpy import mean
from pydantic import BaseModel

from litemind.agent.augmentations.information.information_base import InformationBase
from litemind.agent.messages.message_block import MessageBlock
from litemind.apis.base_api import BaseApi
from litemind.media.media_base import MediaBase
from litemind.utils.pickle_serialisation import PickleSerializable


class Information(InformationBase, PickleSerializable):
    """
    A piece of information that can be stored in a vector database and retrieved for augmentation.

    Informations can have a hierarchical structure with parent-child relationships.
    This is useful for representing complex informations like PDFs with chapters,
    code repositories with files and folders, or any other nested structure.

    Attributes
    ----------
    media: MediaBase
        The media contained in this Information.
    summary: Optional[str]
        A summary of the information's content.
    metadata: Optional[dict]
        Additional metadata about the information.
    _id: Optional[str]
        A unique identifier for the information.
    embedding: Optional[List[float]]
        Vector embedding of the information content.
    _score: Optional[float]
        A relevance score, usually assigned during retrieval.
    parent: Optional[Information]
        The parent information.
    children: List[Information]
        Child informations. Empty list if no children exist.
    """

    def __init__(
        self,
        media: Optional[MediaBase],
        summary: Optional[str] = None,
        metadata: Optional[dict] = None,
        id: Optional[str] = None,
        score: Optional[float] = None,
        parent: Optional["Information"] = None,
        children: Optional[List["Information"]] = None,
    ):
        """
        Create a new Information.

        An Information must have either ``media`` or ``children``, but
        not both. Leaf informations hold media content directly, while
        parent informations aggregate children.

        Parameters
        ----------
        media : Optional[MediaBase]
            The media content. Must be None if ``children`` is provided.
        summary : Optional[str]
            A text summary of the information's content.
        metadata : Optional[dict]
            Additional metadata (e.g., source, author, date).
        id : Optional[str]
            A unique identifier. Auto-generated (UUID4) if not provided.
        score : Optional[float]
            A relevance score, typically assigned during retrieval.
        parent : Optional[Information]
            The parent information in a hierarchy.
        children : Optional[List[Information]]
            Child informations. Must be None if ``media`` is provided.

        Raises
        ------
        ValueError
            If both ``media`` and ``children`` are None, or both are
            provided.
        """

        # Set attributes:
        # If media is None than children must not be None:
        if media is None and children is None:
            raise ValueError(
                "Cannot create an Information with no media and no children."
            )
        # If media is not None than children must be None:
        if media is not None and children is not None:
            raise ValueError(
                "Cannot create an Information with both media and children."
            )

        # Set the attributes:
        self.media = media
        self.children = children if children is not None else []
        self.summary = summary
        self._metadata = metadata if metadata is not None else {}
        self._id = id or str(uuid.uuid4())
        self._score = score
        self.embedding = None
        self.parent = parent

        # Set the parent for each child information:
        for child in self.children:
            child.parent = self

    def has_type(self, media_type: Type[MediaBase]) -> bool:
        return isinstance(self.media, media_type)

    @property
    def type(self) -> Type[MediaBase]:
        return type(self.media)

    @property
    def content(self) -> Any:
        return self.media.get_content()

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def id(self) -> str:
        return self._id

    @property
    def score(self) -> Optional[float]:
        return self._score

    @score.setter
    def score(self, value: Optional[float]):
        self._score = value

    def set_id(self, value: str) -> None:
        """
        Set the ID of this information.

        Parameters
        ----------
        value: str
            The new ID to set.
        """
        self._id = value

    def add_child(self, document: "Information") -> None:
        """
        Add a child information to this information.

        The child's ``parent`` attribute is set to this information.

        Parameters
        ----------
        document : Information
            The child information to add.
        """
        document.parent = self
        self.children.append(document)

    def compute_embedding(
        self, embedding_function: Optional[Callable[[str], List[float]]] = None
    ) -> List[float]:
        """
        Compute an embedding vector for this information.

        For leaf informations, embeds the media content directly. For
        parent informations with children, recursively computes child
        embeddings and averages them. Cached after first computation.

        Parameters
        ----------
        embedding_function : Optional[Callable[[str], List[float]]]
            A function that takes a string and returns an embedding vector.
            If not provided, uses litemind's default API for embeddings.

        Returns
        -------
        List[float]
            The computed embedding vector.

        Raises
        ------
        ValueError
            If the information has no media and no children.
        """
        if embedding_function is None:
            from litemind import CombinedApi

            api = CombinedApi()

            def embedding_function(text):
                return api.embed_texts([text])[0]

        # If we already have an embedding, just return it
        if self.embedding is not None:
            return self.embedding

        # If we don't have children, compute the embedding for our content
        if not self.children:
            if self.media is None:
                raise ValueError(
                    "Cannot compute embedding: Information has no media and no children."
                )
            self.embedding = embedding_function(self.content)
            return list(self.embedding)

        # If we have children, make sure they all have embeddings
        for child in self.children:
            if child.embedding is None:
                child.compute_embedding(embedding_function)

        # Average the embeddings of our children (and own content if we have media)
        embeddings = [child.embedding for child in self.children]
        if self.media is not None:
            embeddings.append(embedding_function(self.content))
        self.embedding = mean(embeddings, axis=0).tolist()

        # Return the computed embedding
        return self.embedding

    def get_all_informations(self) -> List["Information"]:
        """
        Get all informations in the hierarchy as a flat list.

        Returns
        -------
        List[Information]
            A list of all informations in the hierarchy.
        """
        infos = [self]
        for child in self.children:
            infos.extend(child.get_all_informations())
        return infos

    def summarize(
        self,
        summarization_function: Optional[Callable[[str], str]] = None,
        model: Optional[str] = None,
        api: Optional[BaseApi] = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate a text summary for this information.

        Parameters
        ----------
        summarization_function : Optional[Callable[[str], str]]
            A function that takes content and returns a summary string.
            If not provided, uses the LLM API to generate a summary.
        model : Optional[str]
            Model name for LLM-based summarization. Ignored if
            ``summarization_function`` is provided.
        api : Optional[BaseApi]
            API instance for LLM-based summarization. Defaults to
            ``CombinedApi()`` if not provided. Ignored if
            ``summarization_function`` is provided.
        temperature : float
            Sampling temperature for LLM summarization. Ignored if
            ``summarization_function`` is provided.

        Returns
        -------
        str
            The generated summary, also stored in ``self.summary``.

        Raises
        ------
        ValueError
            If ``summarization_function`` is provided together with
            ``api`` or ``model``.
        """
        if summarization_function is None:

            # Use litemind to summarize
            from litemind import CombinedApi
            from litemind.agent.messages.message import Message

            # Create a CombinedApi instance:
            api = api or CombinedApi()

            # Create the system and main messages:
            system_message = Message(
                role="system",
                text="You are a helpful assistant that summarizes informations.",
            )
            main_message = Message(
                role="user",
                text="Please summarize the following document in a concise way.",
            )
            main_message.append_media(self.media)

            # Generate the summary:
            response = api.generate_text(
                messages=[system_message, main_message],
                temperature=temperature,
                model_name=model,
            )

            # Process the response:
            self.summary = response[0].to_plain_text()
        else:
            # Throw an error if api or model are provided:
            if api is not None or model is not None:
                raise ValueError(
                    "Cannot provide api or model if summarization_function is provided."
                )

            self.summary = summarization_function(self.content)

        return self.summary

    def to_message_block(self) -> MessageBlock:
        """
        Convert this information to a MessageBlock for inclusion in a Message.

        Returns
        -------
        MessageBlock
            A message block containing this information's media and metadata.
        """
        return MessageBlock(
            media=self.media,
            metadata=self.metadata,
            id=self.id,
            score=self.score,
        )

    def __copy__(self):
        return Information(
            media=self.media,
            summary=self.summary,
            metadata=self.metadata,
            id=self.id,
            score=self.score,
        )

    def __contains__(self, item):
        return item in self.media

    def __str__(self) -> str:
        return str(self.media)

    def __hash__(self) -> int:
        return hash(self.media)

    def __eq__(self, other) -> bool:
        return self.media == other.media

    def __ne__(self, other) -> bool:
        return not self == other

    def __len__(self) -> int:
        return len(self.media)
