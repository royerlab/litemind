from typing import List, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.conversion.media_converter import MediaConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_image import Image
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video


class MockConverter(BaseConverter):
    def __init__(self, rules):
        self.conversion_rules = rules

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        return self.conversion_rules

    def can_convert(self, media: MediaBase) -> bool:
        return True

    def convert(self, media: MediaBase) -> List[MediaBase]:
        return []


def test_can_convert_within_basic():
    converter = MediaConverter()

    # Create a mock converter with a simple rule
    mock_converter = MockConverter([(Video, [Text, Image, Audio])])
    converter.add_media_converter(mock_converter)

    # Test with allowed set that includes all possible conversions
    assert converter.can_convert_within(Video, {Text, Image, Audio})

    # Test with allowed set missing one conversion result (Audio)
    assert not converter.can_convert_within(Video, {Video, Text})


def test_can_convert_within_multi_step():
    converter = MediaConverter()

    # Create multi-step conversion rules
    converter.add_media_converter(MockConverter([(Video, [Audio, Image])]))
    converter.add_media_converter(MockConverter([(Audio, [Text])]))

    # Test with complete allowed set
    assert converter.can_convert_within(Video, {Video, Audio, Image, Text})

    # Test with incomplete allowed set (missing Text)
    assert not converter.can_convert_within(Video, {Video, Audio, Image})

    # Test with incomplete allowed set (missing Audio)
    assert not converter.can_convert_within(Video, {Video, Image, Text})


def test_can_convert_within_cyclic_conversions():
    converter = MediaConverter()

    # Create cyclic conversion rules
    converter.add_media_converter(MockConverter([(Video, [Audio])]))
    converter.add_media_converter(MockConverter([(Audio, [Image])]))
    converter.add_media_converter(MockConverter([(Image, [Video])]))

    # All types must be in allowed set due to cycle
    assert converter.can_convert_within(Video, {Video, Audio, Image})

    # Missing any type in cycle makes it fail
    assert not converter.can_convert_within(Video, {Video, Audio})
    assert not converter.can_convert_within(Video, {Video, Image})
    assert not converter.can_convert_within(Video, {Audio, Image})


def test_can_convert_within_branching_paths():
    converter = MediaConverter()

    # Create branching conversion paths
    converter.add_media_converter(MockConverter([(Video, [Audio, Image])]))
    converter.add_media_converter(MockConverter([(Audio, [Text])]))
    converter.add_media_converter(MockConverter([(Image, [Table])]))

    # Complete allowed set
    assert converter.can_convert_within(Video, {Video, Audio, Image, Text, Table})

    # Incomplete sets (missing a branch result)
    assert not converter.can_convert_within(Video, {Video, Audio, Image, Text})
    assert not converter.can_convert_within(Video, {Video, Audio, Image, Table})


def test_can_convert_within_empty_converter_list():
    converter = MediaConverter()

    # With empty converter list, only source type is needed in allowed set
    assert converter.can_convert_within(Video, {Video})
    assert converter.can_convert_within(Video, {Video, Text})
    assert not converter.can_convert_within(Video, {Text})


def test_can_convert_within_disjoint_conversion_paths():
    converter = MediaConverter()

    # Set up two disjoint conversion paths
    converter.add_media_converter(MockConverter([(Video, [Audio]), (Image, [Table])]))

    # Test that path from Video only needs to include Video and Audio
    assert converter.can_convert_within(Video, {Video, Audio})

    # Test that path from Image only needs to include Image and Table
    assert converter.can_convert_within(Image, {Image, Table})


def test_can_convert_within_with_default_converters():
    converter = MediaConverter()
    converter.add_default_converters()

    # This test depends on the actual default converters
    # Just ensure the function runs without errors
    result = converter.can_convert_within(Video, {Video, Text, Audio, Image})
    assert isinstance(result, bool)
