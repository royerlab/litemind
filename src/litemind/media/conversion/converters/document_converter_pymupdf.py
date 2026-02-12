"""Document-to-text-and-image converter using PyMuPDF.

Provides :class:`DocumentConverterPymupdf` which extracts text and renders
page images from PDF documents using the PyMuPDF (fitz) library.
"""

from functools import lru_cache
from typing import List, Tuple, Type

from arbol import aprint

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.conversion.converters.document_converter_docling import (
    DocumentConverterDocling,
)
from litemind.media.media_base import MediaBase
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text
from litemind.utils.file_types.file_types import classify_uri
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class DocumentConverterPymupdf(BaseConverter):
    """Converts PDF documents to Text and Image media using PyMuPDF.

    Extracts text and renders a page image for each page of the PDF.
    Only supports PDF files.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Document to Text and Image.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Document to Text and Image.
        """
        return [(Document, [Text, Image])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is a PDF Document.

        Only PDF files are supported by this converter.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a Document with a PDF file type or extension.
        """
        # Check is the media is None:
        if media is None:
            return False

        # Check if the media is a Document:
        if not isinstance(media, Document):
            return False

        extension = media.get_extension()
        file_type = classify_uri(media.uri)

        # Only claim PDF support (matching actual implementation in extract_text_and_image_from_document)
        if "pdf" in file_type or "pdf" in extension:
            return True

        return False

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert a PDF Document to Text and Image media per page.

        Each page produces a Text media with the extracted text and an
        Image media with the rendered page.

        Parameters
        ----------
        media : MediaBase
            The Document media to convert.

        Returns
        -------
        List[MediaBase]
            A preamble Text followed by interleaved Text and Image media
            for each page.

        Raises
        ------
        ValueError
            If *media* is not a Document instance.
        """
        if not isinstance(media, Document):
            raise ValueError(f"Expected Document media, got {type(media)}")

        # Extract text and images from the document:
        text_and_images = extract_text_and_image_from_document(media.uri)

        # Convert these into Text and Image media:
        # Create a list to hold the converted media:
        converted_media = []

        # Get the preamble for the document, we use the same method as in DocumentConverterDocling for consistency:
        preamble = DocumentConverterDocling.get_preamble(media, None)

        # Add a header for the document:
        converted_media.append(Text(preamble))

        for text_media, image_media in text_and_images:
            # Check if the text media is None:
            if text_media is None:
                continue

            # Append the converted media to the list:
            converted_media.append(text_media)

            if image_media is not None:
                converted_media.append(image_media)

        # Return the converted media:
        return converted_media


@lru_cache()
def is_pymupdf_available() -> bool:
    """Check whether PyMuPDF (``pymupdf`` and ``pymupdf4llm``) is installed.

    Returns
    -------
    bool
        True if both packages are available.
    """
    try:
        import importlib.util

        return (
            importlib.util.find_spec("pymupdf4llm") is not None
            and importlib.util.find_spec("pymupdf") is not None
        )

    except ImportError:
        return False


def extract_text_and_image_from_document(
    document_uri: str, dpi: int = 300
) -> List[Tuple[Text, Image]]:
    """Extract text and render an image for each page of a PDF document.

    Parameters
    ----------
    document_uri : str
        The URI or file path of the PDF document.
    dpi : int, optional
        Resolution for the rendered page images. Default is 300.

    Returns
    -------
    List[Tuple[Text, Image]]
        A list of (Text, Image) pairs, one per page. The Text contains
        the extracted page text; the Image is a rendered page image.

    Raises
    ------
    ImportError
        If PyMuPDF is not installed.
    ValueError
        If the document is not a PDF file.
    RuntimeError
        If the document cannot be opened.
    """

    # Check if PyMuPDF is available, if not throw an error!
    if not is_pymupdf_available():
        raise ImportError(
            "pymupdf is not available. Please install it to use this function."
        )

    # Convert the document URI to a local file path.
    document_path = uri_to_local_file_path(document_uri)

    # check if the file is a PDF:
    file_type = classify_uri(document_path)
    if not file_type == "pdf":
        raise ValueError(
            f"Unsupported file type: {file_type}. Only PDF files are supported."
        )

    try:
        # Import the required libraries:
        import fitz  # PyMuPDF
        from pymupdf import Pixmap

        doc = fitz.open(document_path)

        pages_content = []
        # Compute the zoom factor based on the desired DPI (PDFs default to 72 DPI)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_number in range(len(doc)):
            try:
                # Load the page and extract text
                page = doc.load_page(page_number)
                page_text = page.get_text("text")

                # Wrap the page's text in markdown quotes:
                page_text = f"Page {page_number + 1}:\n```text\n{page_text}\n```"

                # Make Text media:
                text_media = Text(page_text)

                # Render the page to an image using the zoom matrix
                pix: Pixmap = page.get_pixmap(matrix=mat)
                pil_image = pix.pil_image()
                image_media = Image.from_PIL_image(pil_image)

                pages_content.append((text_media, image_media))

            except Exception as e:
                aprint(f"Error processing page {page_number}: {e}")
                pages_content.append(
                    (Text(f"Page {page_number} could not be processed."), None)
                )
                continue

        doc.close()

    except Exception as e:
        raise RuntimeError(f"Failed to open document '{document_path}': {e}")

    return pages_content
