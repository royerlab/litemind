from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.conversion.converters.table_converter import TableConverter
from litemind.media.conversion.media_converter import MediaConverter
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.ressources.media_resources import MediaResources


class TestMessageConverter:

    def setup_method(self):
        """Setup for each test."""
        self.converter = MediaConverter()

    def test_init(self):
        """Test the initialization of MediaConverter."""
        assert self.converter.media_converters == []

    def test_add_default_converters(self):
        """Test adding default converters."""
        converters = self.converter.add_default_converters()
        assert len(converters) > 0
        assert len(self.converter.media_converters) > 0
        # Default converters should be the same ones returned
        assert all(c in self.converter.media_converters for c in converters)

    def test_add_remove_converter(self):
        """Test adding and removing converters."""
        # Create a simple converter
        converter = TableConverter()

        self.converter.add_media_converter(converter)
        assert len(self.converter.media_converters) >= 1
        assert converter in self.converter.media_converters

        self.converter.remove_media_converter(converter)
        assert converter not in self.converter.media_converters

    def test_convert_no_conversion_needed(self):
        """Test when all media types are already allowed."""
        # Create messages with different media types
        text_media = Text("Hello world")
        json_media = Json({"name": "test", "value": 42})

        message1 = Message(role="user")
        message1.append_block(MessageBlock(text_media))

        message2 = Message(role="user")
        message2.append_block(MessageBlock(json_media))

        messages = [message1, message2]
        allowed_media_types = [Text, Json]

        # Execute
        result = self.converter.convert(messages, allowed_media_types)

        # Verify
        assert len(result) == 2
        assert result[0] is not message1  # Should be a new message object
        assert isinstance(result[0][0].media, Text)
        assert "Hello world" in result[0][0].media.text
        assert result[1] is not message2
        assert isinstance(result[1][0].media, Json)
        assert result[1][0].media.json == {"name": "test", "value": 42}

    def test_convert_with_default_converters(self):
        """Test conversion using default converters."""
        self.converter.add_default_converters()

        # Create messages with different media types
        table_uri = MediaResources.get_local_test_table_uri("trees.csv")
        table_media = Table(table_uri)

        doc_uri = MediaResources.get_local_test_document_uri("timaeus.txt")
        doc_media = Document(doc_uri)
        json_media = Json({"name": "test", "value": 42})

        message1 = Message(role="user")
        message1.append_block(MessageBlock(table_media))

        message2 = Message(role="user")
        message2.append_block(MessageBlock(doc_media))

        message3 = Message(role="user")
        message3.append_block(MessageBlock(json_media))

        allowed_media_types = [Text]

        # Execute
        result = self.converter.convert(
            [message1, message2, message3], allowed_media_types
        )

        # Verify all were converted to Text
        assert len(result) == 3
        assert all(isinstance(msg[0].media, Text) for msg in result)

    def test_convert_multiple_media_in_message(self):
        """Test conversion when a message has multiple media items."""
        self.converter.add_default_converters()

        doc_uri = MediaResources.get_local_test_document_uri("intracktive_preprint.pdf")

        # Create messages with multiple media items
        text_media = Text("Original text")
        doc_media = Document(doc_uri)
        json_media = Json({"data": "value"})

        message = Message(role="user")
        message.append_block(MessageBlock(text_media))
        message.append_block(MessageBlock(doc_media))
        message.append_block(MessageBlock(json_media))

        # Allow Text only
        allowed_media_types = [Text]

        # Execute
        result = self.converter.convert([message], allowed_media_types)

        # Verify
        assert len(result) == 1
        assert len(result[0].blocks) == 13
        assert all(
            isinstance(block.media, Text) or isinstance(block.media, Image)
            for block in result[0].blocks
        )
        # Original Text should remain unchanged
        assert result[0][0].media.text == "Original text"

    def test_convert_empty_messages(self):
        """Test conversion with empty message list."""
        result = self.converter.convert([], [Text])
        assert result == []

    def test_convert_mixed_allowed_types(self):
        """Test conversion with multiple allowed media types."""
        self.converter.add_default_converters()

        # Create messages with different media types
        table_uri = MediaResources.get_local_test_table_uri("spreadsheet.csv")
        table_media = Table(table_uri)

        doc_uri = MediaResources.get_local_test_document_uri(
            "low_discrepancy_sequence.pdf"
        )
        doc_media = Document(doc_uri)
        json_media = Json({"name": "test", "value": 42})
        text_media = Text("Hello world")

        message1 = Message(role="user")
        message1.append_block(MessageBlock(table_media))

        message2 = Message(role="user")
        message2.append_block(MessageBlock(doc_media))

        message3 = Message(role="user")
        message3.append_block(MessageBlock(json_media))

        message4 = Message(role="user")
        message4.append_block(MessageBlock(text_media))

        # Allow Text and Json
        allowed_media_types = [Text, Json]

        # Execute
        result = self.converter.convert(
            [message1, message2, message3, message4], allowed_media_types
        )

        # Verify
        assert len(result) == 4
        # Table and Document should be converted to Text
        assert isinstance(result[0][0].media, Text)
        assert isinstance(result[1][0].media, Text)
        # Json should stay as Json
        assert isinstance(result[2][0].media, Json)
        # Text should stay as Text
        assert isinstance(result[3][0].media, Text)
        assert result[3][0].media.text == "Hello world"

    def test_complex_message_with_mixed_allowed_types(self):
        """Test conversion of complex messages with multiple allowed types."""
        self.converter.add_default_converters()

        # Create a message with multiple media types
        text_media = Text("Hello")
        json_media = Json({"data": 123})

        table_uri = MediaResources.get_local_test_table_uri("spreadsheet.csv")
        table_media = Table(table_uri)

        doc_uri = MediaResources.get_local_test_document_uri(
            "low_discrepancy_sequence.pdf"
        )
        doc_media = Document(doc_uri)

        message = Message(role="user")
        message.append_block(MessageBlock(text_media))
        message.append_block(MessageBlock(json_media))
        message.append_block(MessageBlock(table_media))
        message.append_block(MessageBlock(doc_media))

        # Allow Text and Json
        allowed_media_types = [Text, Json]

        # Execute
        result = self.converter.convert([message], allowed_media_types)

        # Verify
        assert len(result) == 1
        assert len(result[0].blocks) == 20
        # Text and Json should remain the same
        assert isinstance(result[0][0].media, Text)
        assert isinstance(result[0][1].media, Json)
        # Table and Document should be converted to Text
        assert isinstance(result[0][2].media, Text)
        assert isinstance(result[0][3].media, Text)
        assert isinstance(result[0][4].media, Text)
        assert isinstance(result[0][5].media, Image)
        assert isinstance(result[0][6].media, Text)
        assert isinstance(result[0][7].media, Image)

    def test_all_media_types_conversion(self):
        """Test conversion of all available media types to Text."""
        self.converter.add_default_converters()

        # Create a list of test messages
        test_messages = []

        # Text media
        message = Message(role="user")
        message.append_text("Sample text")
        test_messages.append(message)

        # Json media
        message = Message(role="user")
        message.append_block(MessageBlock(Json({"key": "value"})))
        test_messages.append(message)

        # Table media
        message = Message(role="user")
        table_uri = MediaResources.get_local_test_table_uri("spreadsheet.csv")
        message.append_block(MessageBlock(Table(table_uri)))
        test_messages.append(message)

        # Document media
        message = Message(role="user")
        doc_uri = MediaResources.get_local_test_document_uri(
            "low_discrepancy_sequence.pdf"
        )
        message.append_block(MessageBlock(Document(doc_uri)))
        test_messages.append(message)

        # Test conversion to Text only
        allowed_media_types = [Text]
        result = self.converter.convert(test_messages, allowed_media_types)

        # Verify all were converted to Text
        assert len(result) == len(test_messages)
        assert all(isinstance(msg[0].media, Text) for msg in result)

        # Test with original types allowed
        # Create a message with all media types
        complex_message = Message(role="user")
        complex_message.append_text("Sample text")
        complex_message.append_block(MessageBlock(Json({"key": "value"})))
        complex_message.append_block(MessageBlock(Table(table_uri)))
        complex_message.append_block(MessageBlock(Document(doc_uri)))

        # Allow all original types
        media_types = [Text, Json, Table, Document]
        result = self.converter.convert([complex_message], media_types)

        # Verify all media remain their original type
        assert len(result) == 1
        assert len(result[0].blocks) == 4

        assert isinstance(result[0][0].media, Text)
        assert isinstance(result[0][1].media, Json)
        assert isinstance(result[0][2].media, Table)
        assert isinstance(result[0][3].media, Document)

    def test_get_convertible_media_types(self):
        """Test getting media types that can be converted to allowed media types."""
        self.converter.add_default_converters()

        # Define some allowed media types
        allowed_types = {Text}

        # Get convertable media types
        convertible_types = self.converter.get_convertible_media_types(allowed_types)

        # Ensure we got a set of media types
        assert isinstance(convertible_types, set)

        # Verify that Text itself is in the convertable types
        assert Text in convertible_types

        # Verify that Json is also convertable to Text
        assert Json in convertible_types

        # Test with multiple allowed types
        allowed_types = {Text, Json}
        convertible_types = self.converter.get_convertible_media_types(allowed_types)

        # Both Text and Json should be in the result
        assert Text in convertible_types
        assert Json in convertible_types

        # Table should also be convertable to either Text or Json
        assert Table in convertible_types

    def test_get_convertable_media_types_empty_converters(self):
        """Test getting convertable media types with no converters."""
        # Ensure no converters are present
        self.converter.media_converters = []

        # Only the allowed types themselves should be convertable
        allowed_types = {Text}
        convertable_types = self.converter.get_convertible_media_types(allowed_types)

        # Only Text should be in the result
        assert convertable_types == {Text}

        # Test with multiple allowed types
        allowed_types = {Text, Json}
        convertable_types = self.converter.get_convertible_media_types(allowed_types)

        # Only Text and Json should be in the result
        assert convertable_types == {Text, Json}
