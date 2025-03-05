import copy
from typing import Union

from numpy import ndarray
from pandas import DataFrame
from pydantic import BaseModel

from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.messages.tool_call import ToolCall
from litemind.agent.messages.tool_use import ToolUse


class MessageBlock:
    def __init__(
        self,
        block_type: Union[str, BlockType],
        content: Union[str, BaseModel, ndarray, DataFrame, ToolCall, ToolUse],
        **attributes,
    ):
        """
        Create a new message block.

        Parameters
        ----------
        block_type: Union[str, BlockType]
            The type of the block (e.g., 'text', 'json', 'image', 'audio', 'video', 'document', table, tool ).
        content: Union[str, BaseModel, ndarray, ]
            The content of the block. Can be a free string, a JSON string, a URI, a numpy array, a panda frame, or a Pydantic model.
        """
        if isinstance(block_type, str):
            block_type = BlockType.from_str(block_type)
        self.block_type: BlockType = block_type
        self.content: Union[str, BaseModel, ndarray, DataFrame, ToolCall, ToolUse] = (
            content
        )
        self.attributes = attributes

    def copy(self) -> "MessageBlock":
        """
        Create a copy of the message block.

        Returns
        -------
        MessageBlock
            A copy of the message block.

        """
        return MessageBlock(
            block_type=self.block_type, content=self.content, **self.attributes
        )

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the message block.

        Returns
        -------
        MessageBlock
            A deep copy of the message block.
        """
        return MessageBlock(
            block_type=self.block_type,
            content=copy.deepcopy(self.content, memo),
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
        return text in str(self.content)

    def __str__(self):
        """
        Return the message block as a string.
        Returns
        -------
        str
            The message block as a string.
        """
        if self.block_type == BlockType.Text:
            return self.content
        elif type(self.content) == str:
            return f"{self.block_type}: {self.content}"
        else:
            return f"{type(self.content).__name__}: {self.content}"

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

        if isinstance(self.content, str):
            return len(self.content)
        elif isinstance(self.content, BaseModel):
            return len(self.content.model_dump())
        elif isinstance(self.content, ndarray):
            return self.content.size
        elif isinstance(self.content, DataFrame):
            content: DataFrame = self.content
            return content.size
        elif isinstance(self.content, ToolUse):
            return len(str(self.content.result))
        else:
            return len(str(self.content))
