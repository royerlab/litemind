from typing import List, Sequence, Union

from litemind.agent.messages.message import Message


class Conversation:

    def __init__(self):
        """
        A conversation object that stores messages.

        """

        self.system_messages: List[Message] = []
        self.standard_messages: List[Message] = []

    def get_all_messages(self):
        """
        Get all messages in the conversation.
        Returns
        -------
        List[Message]
            A list of all messages in the conversation

        """
        return self.system_messages + self.standard_messages

    def get_last_message(self):
        """
        Get the last message in the conversation.
        Returns
        -------
        Message
            The last message in the conversation.

        """
        if self.standard_messages:
            return self.standard_messages[-1]
        else:
            return None

    def clear_all(self):
        """
        Clear all messages in the conversation.
        """
        self.system_messages = []
        self.clear_conversation()

    def clear_conversation(self):
        """
        Clear all messages in the conversation.
        """
        self.standard_messages = []

    def append(self, message: Message):
        """
        Append a message to the conversation.
        Parameters
        ----------
        message : Message
            The message to append to the conversation.
        """
        if message.role == "system":
            self.system_messages.append(message)
        else:
            self.standard_messages.append(message)

    def extend(self, messages: Sequence[Message]):
        """
        Extend the conversation with a list of messages.
        Parameters
        ----------
        messages : Sequence[Message]
            The messages to extend the conversation with.
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
        Add two conversations together
        Parameters
        ----------
        other : Conversation
            The conversation to add to the current conversation

        Returns
        -------
        Conversation
            A new conversation object that is the combination of the two conversations

        """
        new_conversation = Conversation()
        new_conversation.system_messages = self.system_messages + other.system_messages
        new_conversation.standard_messages = (
            self.standard_messages + other.standard_messages
        )
        return new_conversation

    def __getitem__(self, item) -> Message:
        """
        Get a message from the conversation by index.
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
        return len(self.get_all_messages())

    def __str__(self):
        return str(self.get_all_messages())

    def __repr__(self):
        return self.__str__()
