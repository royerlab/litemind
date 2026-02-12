"""Default implementation of the ``MediaBase`` interface.

Provides ``MediaDefault``, a concrete base class that implements common
media operations (``to_message_block``, equality, hashing, string
representation) so that concrete media subclasses only need to implement
``get_content``.
"""

from litemind.media.media_base import MediaBase
from litemind.utils.pickle_serialisation import PickleSerializable


class MediaDefault(MediaBase, PickleSerializable):
    """Default implementation of common MediaBase methods.

    Provides standard implementations of ``to_message_block``, ``__str__``,
    ``__len__``, ``__contains__``, ``__hash__``, and ``__eq__`` so that
    concrete media subclasses only need to implement ``get_content``.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword attributes stored in ``self.attributes``.
    """

    def __init__(self, **kwargs):
        self.attributes = kwargs

    def to_message_block(self) -> "MessageBlock":
        """Wrap this media in a MessageBlock.

        Returns
        -------
        MessageBlock
            A new message block containing this media instance.
        """
        from litemind.agent.messages.message_block import MessageBlock

        return MessageBlock(self)

    def __str__(self):
        return str(self.get_content())

    def __len__(self) -> int:
        return len(str(self))

    def __contains__(self, item):
        return item in self.get_content()

    def __hash__(self) -> int:
        return hash(self.get_content())

    def __eq__(self, other):
        if not isinstance(other, MediaDefault):
            return False
        return self.get_content() == other.get_content()
