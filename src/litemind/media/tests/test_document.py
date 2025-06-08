import os
import tempfile

import pytest

from litemind.media.media_uri import MediaURI
from litemind.media.types.media_document import Document


class TestDocument:

    def test_document_initialization_with_valid_local_file(self):
        """Test basic initialization with a valid local file URI."""

        # Create a temporary PDF file for testing in a temporary directory using tempfile:
        path = os.path.join(tempfile.gettempdir(), "example.pdf")

        # Create a dummy pdf file
        with open(path, "w") as f:
            f.write("Dummy PDF content")

        uri = "file://" + path
        document = Document(uri=uri)
        assert document.uri == uri
        assert document.extension == None
        os.remove(path)

    def test_document_initialization_with_valid_local_file_and_extension(self):
        """Test basic initialization with a valid local file URI and extension."""

        # Create a temporary PDF file for testing in a temporary directory using tempfile:
        path = os.path.join(tempfile.gettempdir(), "example2.pdf")

        # Create a dummy pdf file
        with open(path, "w") as f:
            f.write("Dummy PDF content")

        uri = "file://" + path
        document = Document(uri=uri, extension="pdf")
        assert document.uri == uri
        assert document.extension == "pdf"
        os.remove(path)

    def test_document_initialization_with_valid_remote_url(self):
        """Test basic initialization with a valid remote URL."""
        uri = "http://example.com/example.pdf"
        document = Document(uri=uri)
        assert document.uri == uri
        assert document.extension == None

    def test_document_initialization_with_invalid_local_file_extension(self):
        """Test initialization with an invalid local file extension raises ValueError."""
        uri = "example.sdse"
        with pytest.raises(
            ValueError,
            match="Invalid URI or local file path: 'example.sdse'.",
        ):
            Document(uri=uri)

    def test_document_initialization_with_no_extension_remote_url(self):
        """Test initialization with a remote URL without extension."""
        uri = "http://example.com/example"
        document = Document(uri=uri)
        assert document.uri == uri
        assert document.extension == None

    def test_document_initialization_with_upper_case_extension(self):
        """Test initialization with upper case extension."""
        uri = "EXAMPLE.PDF"  # Assuming 'example.pdf' exists in the same directory
        with pytest.raises(
            ValueError,
            match="Invalid URI or local file path: 'EXAMPLE.PDF'.",
        ):
            Document(uri=uri)

    def test_document_initialization_with_mixed_case_extension(self):
        """Test initialization with mixed case extension."""
        uri = "ExAmPlE.pDf"  # Assuming 'example.pdf' exists in the same directory
        with pytest.raises(
            ValueError,
            match="Invalid URI or local file path: 'ExAmPlE.pDf'.",
        ):
            Document(uri=uri)

    def test_inheritance(self):
        """Test that Document inherits from MediaURI."""
        path = "example.pdf"  # Assuming 'example.pdf' exists in the same directory
        # Create a dummy pdf file
        with open(path, "w") as f:
            f.write("Dummy PDF content")

        uri = "file://" + path
        document = Document(uri=uri)
        assert isinstance(document, MediaURI)
        os.remove(path)

    def test_extract_text_from_pages(self):
        """Test the extract_text_from_pages method."""

        # Create a temporary PDF file for testing in a temporary directory using tempfile:
        path = os.path.join(tempfile.gettempdir(), "test_document.pdf")

        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=216, height=720)
        with open(path, "wb") as f:
            writer.write(f)

        uri = "file://" + path
        document = Document(uri=uri)
        text_pages = document.extract_text_from_pages()
        assert isinstance(text_pages, list)
        assert len(text_pages) > 0
        os.remove(path)

    def test_take_image_of_each_page(self):
        """Test the take_image_of_each_page method."""

        # Create a temporary PDF file for testing in a temporary directory using tempfile:
        path = os.path.join(tempfile.gettempdir(), "test_document_2.pdf")

        from pypdf import PdfWriter

        writer = PdfWriter()
        writer.add_blank_page(width=216, height=720)
        with open(path, "wb") as f:
            writer.write(f)

        uri = "file://" + path
        document = Document(uri=uri)
        image_pages = document.take_image_of_each_page()
        assert isinstance(image_pages, list)
        assert len(image_pages) > 0
        # Basic check to see if the returned items are valid image file paths or image objects
        for img_path in image_pages:
            # remove the file:// prefix if it exists:
            if img_path.startswith("file://"):
                img_path = img_path[7:]
            assert os.path.exists(img_path)
            os.remove(img_path)
        os.remove(path)
