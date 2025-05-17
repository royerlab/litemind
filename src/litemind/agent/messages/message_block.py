import copy
from typing import Any, Type

from litemind.media.media_base import MediaBase
from litemind.media.types.media_object import Object
from litemind.media.types.media_text import Text


class MessageBlock:
    """
    Message blocks are the building blocks of messages.
    Each message block is a container for media content and its associated attributes.
    """

    def __init__(
        self,
        media: MediaBase,
        **attributes,
    ):
        """
        Create a new message block.

        Parameters
        ----------
        media: MediaBase
            The media content of the block. Can be a free string, a JSON string, a URI, a numpy array, a panda frame, or a Pydantic model.
        """
        self.media: MediaBase = media
        self.attributes: dict = attributes

    def get_content(self) -> Any:
        """
        Get the content of the message block.

        Returns
        -------
        Any
            The content of the message block.
        """
        return self.media.get_content()

    def has_type(self, block_type: Type[MediaBase]) -> bool:
        """
        Returns True if the message block has the specified media type.

        Parameters
        ----------
        block_type: Type[MediaBase]
            The media type to check.

        Returns
        -------
        bool
        True if this message block has the specified media type.

        """
        return isinstance(self.media, block_type)

    def get_type(self) -> Type[MediaBase]:
        """
        Returns the message block type.

        Returns
        -------
        Type[MediaBase]
        The message block type.

        """

        return type(self.media)

    def get_type_name(self) -> str:
        """
        Returns the message block type name.

        Returns
        -------
        str
            The message block type name.
        """
        return self.get_type().__name__

    def has_attribute(self, attribute_key):
        """
        Check if the message block has the specified attribute.

        Parameters
        ----------
        attribute_key: str
            The attribute key to check.

        Returns
        -------

        """
        return attribute_key in self.attributes

    def is_thinking(self) -> bool:
        """
        Check if the message block is a thinking block.

        Returns
        -------
        bool
            True if the message block is a thinking block, False otherwise.
        """
        return self.has_attribute("thinking")

    def is_redacted(self) -> bool:
        """
        Check if the message block is redacted.

        Returns
        -------
        bool
            True if the message block is redacted, False otherwise.
        """
        return self.has_attribute("redacted")

    def copy(self) -> "MessageBlock":
        """
        Create a copy of the message block.

        Returns
        -------
        MessageBlock
            A copy of the message block.

        """
        return MessageBlock(media=self.media, **self.attributes)

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the message block.

        Returns
        -------
        MessageBlock
            A deep copy of the message block.
        """
        return MessageBlock(
            media=copy.deepcopy(self.media, memo),
            **copy.deepcopy(self.attributes, memo),
        )

    def contains(self, text: str) -> bool:
        """
        Check if the message block contains the given text.

        Parameters
        ----------
        text : str
            The text to search for.

        Returns
        -------
        bool
            True if the text is found in the message block, False otherwise.
        """
        return text in str(self.get_content())

    def __str__(self):
        """
        Return the message block as a string.

        Returns
        -------
        str
            The message block as a string.
        """
        if isinstance(self.media, Text):
            return self.media.text
        elif "thinking" in self.attributes:
            return f"<thinking>\n{self.get_content().strip()}\n<thinking/>\n"
        elif self.has_type(Object):
            return f"{type(self.get_content()).__name__}: {self.get_content()}"
        else:
            return f"{type(self.media).__name__}: {str(self.media)}"

    def __repr__(self):
        """
        Return the message block as a string.
        Returns
        -------
        str
            The message block as a string.
        """
        return str(self)

    def __len__(self) -> int:
        """
        Get the length of the content of the message block.
        Returns
        -------
        int
            The length of the content of the message block.
        """

        return len(self.media)
