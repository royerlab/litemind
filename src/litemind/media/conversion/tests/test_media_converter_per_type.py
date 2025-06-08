from pandas import DataFrame
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.conversion.media_converter import MediaConverter
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_code import (
    Code,
)  # Assuming this is the import path for Code media
from litemind.media.types.media_document import Document
from litemind.media.types.media_file import File
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_ndimage import NdImage
from litemind.media.types.media_object import Object
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.ressources.media_resources import MediaResources


class TestMessageConverterPerType:

    def setup_method(self):
        """Setup for each test."""
        self.converter = MediaConverter()

    def test_convert_audio_media(self):
        """Test conversion of audio media."""
        self.converter.add_default_converters()

        # Create message with audio media
        audio_uri = MediaResources.get_local_test_audio_uri("zebrahub_short.mp3")
        audio_media = Audio(audio_uri)

        message = Message(role="user")
        message.append_block(MessageBlock(audio_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)
        # Audio typically converts to transcript text
        assert len(result[0][0].media.text) > 0

        # Extract text:
        transcription = result[0][0].media.text.lower()

        # Basic check that some content was extracted
        assert isinstance(transcription, str)

        # Check that 'zebrahub' is in the text:
        assert "embryo" in transcription
        assert "development" in transcription
        assert "zebrafish" in transcription

    def test_convert_code_media(self):
        """Test conversion of code media."""
        self.converter.add_default_converters()

        # Create message with code media
        code_uri = MediaResources.get_local_test_document_uri("fib.cpp")

        # Remove 'file://' ffrom the URI:
        code_path = code_uri.replace("file://", "")

        # Get file contents:
        with open(code_path, "r") as file:
            code_content = file.read()

        code_media = Code(code_content, lang="cpp")  # Ensure Code class is imported

        message = Message(role="user")
        message.append_block(MessageBlock(code_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        code_text = result[0][0].media.text.lower()

        # Basic check that some content was extracted
        assert isinstance(code_text, str)
        assert len(code_text) > 0

        # Check for typical content in your fib.cpp
        assert "fibonacci" in code_text
        assert "recursion" in code_text
        assert "fib(n - 1)" in code_text
        assert "#include <bits/stdc++.h>" in code_text
        assert "using namespace std;" in code_text
        assert "int main()" in code_text
        assert "return 0;" in code_text

    def test_convert_docx_document_media(self):
        """Test conversion of DOCX documents media."""
        self.converter.add_default_converters()

        # Create message with docx media
        docx_uri = MediaResources.get_local_test_document_uri("maya_takahashi_cv.docx")
        docx_media = Document(docx_uri)

        message = Message(role="user")
        message.append_block(MessageBlock(docx_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        docx_text = str(result).lower()

        # Verify basic content characteristics - adjust based on your sample docx
        assert "skills" in docx_text
        assert "(206)-555-7890" in docx_text
        assert "maya takahashi" in docx_text
        assert "cv" in docx_text
        assert "lymphocyte" in docx_text

    def test_convert_html_document_media(self):
        """Test conversion of HTML document media."""
        self.converter.add_default_converters()

        # Create message with html media
        html_uri = MediaResources.get_local_test_document_uri("sample.html")
        html_media = Document(html_uri)

        message = Message(role="user")
        message.append_block(MessageBlock(html_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)
        # Verify HTML content was properly converted to markdown
        assert len(result[0][0].media.text) > 0

        # Extract text:
        html_text = str(result).lower()

        # Look for typical HTML to markdown conversion artifacts
        assert "---" in html_text  # From the file metadata header

        # make sure the following words are present: heading, ipsum, #include,  explanation, recording:
        assert "heading" in html_text
        assert "ipsum" in html_text
        assert "#include" in html_text
        assert "explanation" in html_text
        assert "recording" in html_text

    def test_convert_pdf_document_media(self):
        """Test conversion of PDF document media."""
        self.converter.add_default_converters()

        # Create message with PDF media
        pdf_uri = MediaResources.get_local_test_document_uri("intracktive_preprint.pdf")
        pdf_media = Document(pdf_uri)

        message = Message(role="user")
        message.append_block(MessageBlock(pdf_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        pdf_text = str(result[0])

        # Basic check that some content was extracted
        assert isinstance(pdf_text, str)
        assert len(pdf_text) > 0

        # Check for typical content in your PDF
        # Adjust these based on your sample.pdf content
        assert "cell" in pdf_text
        assert "tracking" in pdf_text
        assert "intracktive" in pdf_text

    def test_convert_file_media(self):
        """Test conversion of a generic file media."""
        self.converter.add_default_converters()

        # Create message with file media
        file_uri = MediaResources.get_local_test_other_uri("file.dat")

        # Create a file media object
        file_media = File(file_uri)  # Assuming File(uri) constructor

        # Create a message and append the file media
        message = Message(role="user")
        message.append_block(MessageBlock(file_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        file_text = result[0][0].media.text

        # Basic check that some content was extracted
        assert isinstance(file_text, str)
        assert len(file_text) > 0

        assert "Description of file: 'file.dat'" in file_text
        assert "fb 00 00 00 00" in file_text
        assert "File Size: 6032 bytes (5.89 KB)" in file_text

    def test_convert_image_media(self):
        """Test conversion of image media."""
        self.converter.add_default_converters()

        # Create message with image media
        image_uri = MediaResources.get_local_test_image_uri("cat.jpg")
        image_media = Image(image_uri)

        message = Message(role="user")
        message.append_block(MessageBlock(image_media))

        # Convert to Text
        result = self.converter.convert([message], [Text, Image])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Image)

        # Extract text:
        image_uri = result[0][0].media.uri

        # Basic check that some content was extracted
        assert isinstance(image_uri, str)
        assert len(image_uri) > 0

        # Check for typical image description keywords
        # Adjust based on the expected output of your image-to-text conversion
        assert "cat" in image_uri

    def test_convert_json_media(self):
        """Test conversion of JSON media."""

        self.converter.add_default_converters()

        # Create message with JSON media
        json_data = {"name": "test", "values": [1, 2, 3], "nested": {"key": "value"}}
        json_media = Json(json_data)

        message = Message(role="user")
        message.append_block(MessageBlock(json_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        json_text = result[0][0].media.text.lower()

        # Basic check that some content was extracted
        assert isinstance(json_text, str)
        assert len(json_text) > 0

        # Check for JSON content in markdown format
        assert "name" in json_text
        assert "test" in json_text
        assert "values" in json_text
        assert "nested" in json_text
        assert "key" in json_text
        assert "value" in json_text

    def test_convert_ndimage_media(self):
        """Test conversion of NDImage media."""
        self.converter.add_default_converters()

        # Create message with NDImage media
        # Create a sample numpy array
        ndimage_uri = MediaResources.get_local_test_ndimage_uri("tubhiswt_C1.ome.tif")
        ndimage_media = NdImage(ndimage_uri)

        message = Message(role="user")
        message.append_block(MessageBlock(ndimage_media))

        # Convert to Text
        # NDImage typically converts to a textual description or potentially an Image if visualizable
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        ndimage_text = str(result)

        # Basic check that some content was extracted
        assert isinstance(ndimage_text, str)
        assert len(ndimage_text) > 0

        # Check for typical content in the textual representation of an NDImage
        # This will depend on how NDImage.to_markdown_text_media() is implemented
        assert "## nD Image 'tubhiswt_C1.ome.tif'" in ndimage_text
        assert "**Spatial dimensions:** 1, 2, 3" in ndimage_text
        assert "**Dimensions:** 2 × 20 × 512 × 512" in ndimage_text
        assert "**Data type:** uint8" in ndimage_text
        assert "### Channel: x=1" in ndimage_text
        assert "### Maximum Intensity Projection: zt-plane" in ndimage_text
        assert "**Value range:** [0, 215]" in ndimage_text

    def test_convert_object_media(self):
        """Test conversion of Object media."""
        self.converter.add_default_converters()

        # A simple class for testing Object media conversion
        class SampleObjectForTest(BaseModel):
            name: str
            value: int
            description: str = "This is a sample object for testing."

            def __str__(self):
                return f"SampleObjectForTest(name='{self.name}', value={self.value}, description='{self.description}')"

            def __repr__(self):
                return self.__str__()

        # Create message with Object media
        sample_obj = SampleObjectForTest(name="TestObject", value=123)
        object_media = Object(sample_obj)

        message = Message(role="user")
        message.append_block(MessageBlock(object_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        object_text = str(result[0][0])

        # Basic check that some content was extracted
        assert isinstance(object_text, str)
        assert len(object_text) > 0

        # Check for typical content in the textual representation of an Object
        # This will depend on how Object.to_markdown_text_media() is implemented
        # and the __str__ or __repr__ of the SampleObjectForTest
        assert "```json" in object_text
        assert "\n```" in object_text
        assert "TestObject" in object_text
        assert "This is a sample object for testing" in object_text
        assert ":123" in object_text

    def test_convert_table_media(self):
        """Test conversion of Table media."""
        self.converter.add_default_converters()

        # Create message with Table media
        data = {
            "Name": ["Alice", "Bob", "Charlie"],
            "Age": [30, 24, 35],
            "City": ["New York", "San Francisco", "London"],
        }

        # Create a Table object
        table_media = Table.from_dataframe(DataFrame(data))

        # Create a message and append the table media
        message = Message(role="user")
        message.append_block(MessageBlock(table_media))

        # Convert to Text
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        table_text = result[0][0].media.text

        # Basic check that some content was extracted
        assert isinstance(table_text, str)
        assert len(table_text) > 0

        # Check for typical content in the textual representation of a Table (e.g., Markdown)
        assert "Name" in table_text  # Check if the table name is present
        assert "Alice" in table_text  # Check for header row
        assert "24" in table_text  # Check for markdown table separator
        assert "San Francisco" in table_text  # Check for data row 1

    def test_convert_text_media(self):
        """Test conversion of Text media (identity conversion)."""
        self.converter.add_default_converters()

        # Create message with Text media
        original_text_content = "This is a sample text for testing conversion."
        text_media = Text(text=original_text_content)

        message = Message(role="user")
        message.append_block(MessageBlock(text_media))

        # Convert to Text (should be an identity conversion)
        result = self.converter.convert([message], [Text])

        # Verify conversion results
        assert len(result) == 1
        assert isinstance(result[0][0].media, Text)

        # Extract text:
        converted_text = result[0][0].media.text

        # Basic check that some content was extracted
        assert isinstance(converted_text, str)
        assert len(converted_text) > 0

        # Check that the text content is unchanged
        assert converted_text == original_text_content

    def test_convert_video_media(self):
        """Test conversion of video media."""

        # Initialize the converter
        self.converter.add_default_converters()

        # Create message with video media
        video_uri = MediaResources.get_local_test_video_uri("job_interview.mp4")
        video_media = Video(video_uri)

        message = Message(role="user")
        message.append_block(MessageBlock(video_media))

        # Convert to Text
        result = self.converter.convert([message], [Text, Image])

        # Verify conversion results
        assert len(result) == 1
        # Video conversion might produce multiple blocks (transcript and frames)
        assert len(result[0].blocks) >= 100
        # First block is typically the metadat info:
        assert isinstance(result[0][0].media, Text)
        # All blocks should be either Text or Image:
        assert all(
            isinstance(block.media, Text) or isinstance(block.media, Image)
            for block in result[0].blocks
        )

        # Extract transcript text:
        transcript = result[0][-1].media.text.lower()

        # The last block should be the audio transcript, check if it contains: job, glimmer, florida:
        assert "job" in transcript
        assert "glimmer" in transcript
        assert "florida" in transcript
