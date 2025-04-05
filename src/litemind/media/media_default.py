from litemind.media.media_base import MediaBase
from litemind.utils.pickle_serialisation import PickleSerializable


class MediaDefault(MediaBase, PickleSerializable):
    """
    This class provides default and common implementations of methods for MediaBase

    """

    def __init__(self, **kwargs):
        self.attributes = kwargs

    def to_message_block(self) -> "MessageBlock":
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
