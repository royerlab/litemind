"""Conversation module for managing ordered sequences of messages.

A Conversation stores system messages separately from standard
(user/assistant/tool) messages so that system messages can be
consistently placed at the beginning of the list sent to the LLM API.
"""

from typing import List, Sequence, Union

from litemind.agent.messages.message import Message


class Conversation:
    """A conversation that stores system and standard messages separately.

    System messages are kept separate from standard (user/assistant/tool)
    messages so they can be consistently placed at the beginning of the
    message list sent to the LLM API.
    """

    def __init__(self):
        """Create a new empty conversation."""

        self.system_messages: List[Message] = []
        self.standard_messages: List[Message] = []

    def get_all_messages(self):
        """
        Get all messages in the conversation.

        Returns
        -------
        List[Message]
            A list of all messages, with system messages first.
        """
        return self.system_messages + self.standard_messages

    def get_last_message(self):
        """
        Get the last standard (non-system) message in the conversation.

        Returns
        -------
        Message or None
            The last standard message, or None if there are no standard
            messages.
        """
        if self.standard_messages:
            return self.standard_messages[-1]
        else:
            return None

    def clear_all(self):
        """
        Clear both system and standard messages in the conversation.
        """
        self.system_messages = []
        self.clear_standard_messages()

    def clear_standard_messages(self):
        """
        Clear standard (non-system) messages in the conversation.
        """
        self.standard_messages = []

    # Backwards compatibility alias
    def clear_conversation(self):
        """
        Clear standard messages in the conversation.

        .. deprecated::
            Use :meth:`clear_standard_messages` instead.
        """
        self.clear_standard_messages()

    def append(self, message: Message):
        """
        Append a message to the conversation.

        System messages are routed to ``system_messages``; all others go
        to ``standard_messages``.

        Parameters
        ----------
        message : Message
            The message to append.
        """
        if message.role == "system":
            self.system_messages.append(message)
        else:
            self.standard_messages.append(message)

    def extend(self, messages: Sequence[Message]):
        """
        Extend the conversation with multiple messages.

        Parameters
        ----------
        messages : Sequence[Message]
            The messages to append. Each is routed by its role.
        """
        for message in messages:
            self.append(message)

    def __iadd__(self, other: Union["Conversation", Message]):
        """
        Add a message or conversation to the conversation.

        Parameters
        ----------
        other : Union['Conversation', Message]
            The message or conversation to add to the conversation.

        """
        if isinstance(other, Message):
            self.append(other)
            return self
        elif isinstance(other, Conversation):
            self.system_messages.extend(other.system_messages)
            self.standard_messages.extend(other.standard_messages)
        else:
            raise ValueError("Can only add Conversation or Message objects")
        return self

    def __add__(self, other: "Conversation"):
        """
        Combine two conversations into a new one.

        Parameters
        ----------
        other : Conversation
            The conversation to combine with.

        Returns
        -------
        Conversation
            A new conversation containing messages from both conversations.
        """
        new_conversation = Conversation()
        new_conversation.system_messages = self.system_messages + other.system_messages
        new_conversation.standard_messages = (
            self.standard_messages + other.standard_messages
        )
        return new_conversation

    def __getitem__(self, item) -> Message:
        """
        Get a message by index across all messages.

        Parameters
        ----------
        item : int
            The index of the message to get.

        Returns
        -------
        Message
            The message at the specified index.
        """
        return self.get_all_messages()[item]

    def __len__(self):
        """Return the total number of messages in the conversation.

        Returns
        -------
        int
            The combined count of system and standard messages.
        """
        return len(self.get_all_messages())

    def __str__(self):
        """Return a string representation of all messages in the conversation.

        Returns
        -------
        str
            A string representation of the list of all messages, with
            system messages first followed by standard messages.
        """
        return str(self.get_all_messages())

    def __repr__(self):
        """Return the string representation of the conversation.

        Returns
        -------
        str
            Same as ``__str__``.
        """
        return self.__str__()
