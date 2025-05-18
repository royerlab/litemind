from typing import Any, List, Tuple, Type

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.conversion.media_converter import MediaConverter
from litemind.media.media_base import MediaBase
from litemind.media.media_default import MediaDefault
from litemind.media.types.media_text import Text


class UnsupportedMedia(MediaDefault):
    """Custom media type with no available converter."""

    def __init__(self, data):
        self.data = data

    def get_extension(self):
        return "unsupported"

    def get_content(self) -> Any:
        return self.data


class IntermediateMedia(MediaDefault):
    """Media type that requires multiple conversion steps."""

    def __init__(self, data):
        self.data = data

    def get_extension(self):
        return "intermediate"

    def get_content(self) -> Any:
        return self.data


class FailingMedia(MediaDefault):
    """Media type that will trigger converter failure."""

    def __init__(self):
        pass

    def get_extension(self):
        return "failing"

    def get_content(self) -> Any:
        return self.data


class IntermediateConverter(BaseConverter):
    """Converter that converts UnsupportedMedia to IntermediateMedia."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(UnsupportedMedia, [IntermediateMedia])]

    def can_convert(self, media):
        return isinstance(media, UnsupportedMedia)

    def convert(self, media):
        return [IntermediateMedia(f"Converted from {media.data}")]


class IntermediateToTextConverter(BaseConverter):
    """Converter that converts IntermediateMedia to Text."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(IntermediateMedia, [Text])]

    def can_convert(self, media):
        return isinstance(media, IntermediateMedia)

    def convert(self, media):
        return [Text(f"Final conversion: {media.data}")]


class FailingConverter(BaseConverter):
    """Converter that raises an exception during conversion."""

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return [(Text, [Text])]

    def can_convert(self, media):
        return isinstance(media, FailingMedia)

    def convert(self, media):
        raise RuntimeError("Conversion failed!")


class TestMessageConverterExtra:

    def setup_method(self):
        """Setup for each test."""
        self.converter = MediaConverter()

    def test_recursive_conversion(self):
        """Test recursive conversion of media types."""
        # Add converters that require multiple steps
        intermediate_converter = IntermediateConverter()
        intermediate_to_text_converter = IntermediateToTextConverter()
        self.converter.add_media_converter(intermediate_converter)
        self.converter.add_media_converter(intermediate_to_text_converter)

        # Create message with media that needs multiple conversion steps
        message = Message(role="user")
        message.append_block(MessageBlock(UnsupportedMedia("test_data")))

        # Without recursive flag, it should only convert once
        result = self.converter.convert([message], [Text], recursive=False)
        assert len(result) == 1
        assert isinstance(result[0][0].media, IntermediateMedia)

        # With recursive flag, it should convert all the way to Text
        result = self.converter.convert([message], [Text], recursive=True)
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)
        assert "Final conversion: Converted from test_data" in result[0][0].media.text

    def test_unsupported_media_type(self):
        """Test handling of media types with no available converter."""
        # Create a message with an unsupported media type
        message = Message(role="user")
        message.append_block(MessageBlock(UnsupportedMedia("no_converter_available")))

        # No converters for UnsupportedMedia
        result = self.converter.convert([message], [Text])

        # Should keep the original media
        assert len(result) == 1
        assert isinstance(result[0][0].media, UnsupportedMedia)
        assert result[0][0].media.data == "no_converter_available"

    def test_failing_converter(self):
        """Test handling of converters that raise exceptions."""
        # Add a failing converter
        failing_converter = FailingConverter()
        self.converter.add_media_converter(failing_converter)

        # Create message with media that will cause conversion failure
        message = Message(role="user")
        message.append_block(MessageBlock(FailingMedia()))

        # Convert should handle the exception and keep the original media
        result = self.converter.convert([message], [Text])

        # Verify the message structure is preserved and original media remains
        assert len(result) == 1
        assert len(result[0].blocks) == 2
        assert isinstance(result[0][0].media, Text)
        assert isinstance(result[0][1].media, FailingMedia)

        # The converter should have attempted conversion but failed gracefully
        assert message[0] == result[0][1]  # Original message should be unchanged
