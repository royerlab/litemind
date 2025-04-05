from typing import Any

import pytest

from litemind.agent.messages.message_block import MessageBlock
from litemind.media.media_base import MediaBase


class TestMediaBase:
    def test_initialization(self):
        """Test that MediaBase can be initialized."""

        class ConcreteMedia(MediaBase):

            def get_content(self) -> Any:
                return "Concrete content"

            def __str__(self):
                return "ConcreteMedia"

            def __len__(self):
                return 0

            def to_message_block(self) -> "MessageBlock":
                return MessageBlock(media=self)

        media = ConcreteMedia()
        assert isinstance(media, MediaBase)

    def test_abstract_methods(self):
        """Test that MediaBase requires implementation of abstract methods."""
        with pytest.raises(TypeError):
            MediaBase()
