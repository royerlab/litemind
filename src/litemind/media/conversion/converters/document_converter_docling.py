"""Document-to-text converter using the Docling library.

Provides :class:`DocumentConverterDocling` which converts PDF, DOCX, XLSX,
PPTX, Markdown, AsciiDoc, and HTML documents into per-page Text media using
the Docling document conversion library.
"""

from functools import lru_cache
from typing import List, Optional, Tuple, Type

from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_document import Document
from litemind.media.types.media_text import Text
from litemind.utils.file_types.file_types import classify_uri
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class DocumentConverterDocling(BaseConverter):
    """Converts documents to Text media using the Docling library.

    Supports PDF, DOCX, XLSX, PPTX, Markdown, AsciiDoc, and HTML files.
    Produces one Text media per page of the converted document.
    """

    def rule(self) -> List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]:
        """Declare that this converter transforms Document to Text.

        Returns
        -------
        List[Tuple[Type[MediaBase], List[Type[MediaBase]]]]
            A single rule mapping Document to Text.
        """
        return [(Document, [Text])]

    def can_convert(self, media: MediaBase) -> bool:
        """Check whether the given media is a supported Document type.

        Supports PDF, DOCX, XLSX, PPTX, Markdown, AsciiDoc, HTML, and
        XHTML files.

        Parameters
        ----------
        media : MediaBase
            The media to check.

        Returns
        -------
        bool
            True if the media is a Document with a supported file type.
        """
        # Check is the media is None:
        if media is None:
            return False

        # Check if the media is a Document:
        if not isinstance(media, Document):
            return False

        extension = media.get_extension()
        file_type = classify_uri(media.uri)

        if "pdf" in file_type or "pdf" in extension:
            return True

        if "office" in file_type and "docx" in extension:
            return True

        if "office" in file_type and "xlsx" in extension:
            return True

        if "office" in file_type and "pptx" in extension:
            return True

        if "text" in file_type and "md" in extension:
            return True

        if "text" in file_type and "adoc" in extension:
            return True

        if "text" in file_type and ("html" in extension or "xhtml" in extension):
            return True

        return False

    def convert(self, media: MediaBase) -> List[MediaBase]:
        """Convert a Document to a list of per-page Text media using Docling.

        Parameters
        ----------
        media : MediaBase
            The Document media to convert.

        Returns
        -------
        List[MediaBase]
            A list of Text media: a preamble header followed by one Text
            per page of the document.

        Raises
        ------
        ValueError
            If *media* is not a Document instance.
        """
        if not isinstance(media, Document):
            raise ValueError(f"Expected Document media, got {type(media)}")

        # Extract text pages:
        pages_text = convert_to_markdown(media.uri)

        # Create a list to hold the converted media:
        converted_media = []

        # Number of pages:
        num_pages = len(pages_text)

        # Get the preamble for the document:
        preamble = DocumentConverterDocling.get_preamble(media, num_pages)

        # Add a header for the document:
        converted_media.append(Text(preamble))

        # Convert the pages into Text media:
        if len(pages_text) == 1:
            # If there is only one page, we append it:
            converted_media.append(Text(pages_text[0]))

        else:
            for page_text in pages_text:
                # add preamble that specifies the page in markdown compatible format:
                page_text = f"---\nPage {pages_text.index(page_text) + 1} of {len(pages_text)}\n---\n{page_text}"

                # Append the converted media to the list:
                converted_media.append(Text(page_text))

        # Return the converted media:
        return converted_media

    @staticmethod
    def get_preamble(media, num_pages: Optional[int] = None):
        """Build a descriptive header string for a document.

        Parameters
        ----------
        media : Document
            The document media to describe.
        num_pages : int or None, optional
            The number of pages. Displayed as ``"unknown"`` if None.

        Returns
        -------
        str
            A multi-line string with filename, type, extension, and page count.
        """
        # Get the file name:
        file_name = media.get_filename()
        # Get the file extension:
        file_extension = media.get_extension()
        # Get the file type:
        file_type = classify_uri(media.uri)

        # Deal with the number of pages is not available:
        if num_pages is None:
            num_pages = "unknown"
        else:
            num_pages = int(num_pages)

        # Make a preamble for all pages that gives the file name and the number of pages:
        preamble = f"\nDocument: {file_name}\nType: {file_type}\nExtension: {file_extension}\nNumber of pages: {num_pages} \n "

        return preamble


@lru_cache()
def is_docling_available() -> bool:
    """Check whether the ``docling`` library is installed.

    Returns
    -------
    bool
        True if docling can be found by importlib.
    """
    try:
        import importlib.util

        return importlib.util.find_spec("docling") is not None

    except ImportError:
        return False


def initialize_docling_converter():
    """Create and configure a Docling DocumentConverter.

    Enables OCR, table extraction, and page image generation for PDFs,
    and page image generation for DOCX files.

    Returns
    -------
    docling.document_converter.DocumentConverter
        A configured document converter instance.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PaginatedPipelineOptions,
        PdfPipelineOptions,
    )
    from docling.document_converter import (
        DocumentConverter,
        PdfFormatOption,
        WordFormatOption,
    )

    # Configure pipeline options for PDF
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.do_ocr = True  # Enable OCR for image-based PDFs
    pdf_pipeline_options.do_table_structure = True  # Extract tables
    pdf_pipeline_options.generate_page_images = True  # Save page images

    # Configure pipeline options for DOCX
    docx_pipeline_options = PaginatedPipelineOptions()
    docx_pipeline_options.generate_page_images = True  # Save page images

    # Create document converter with our configured options for both formats
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
            InputFormat.DOCX: WordFormatOption(pipeline_options=docx_pipeline_options),
        }
    )

    return converter


# Lazy initialization - only create converter when actually needed
__default_docling_converter = None


def _get_docling_converter():
    """Get or lazily initialise the singleton Docling converter.

    Returns
    -------
    docling.document_converter.DocumentConverter
        The shared Docling converter instance.
    """
    global __default_docling_converter
    if __default_docling_converter is None:
        __default_docling_converter = initialize_docling_converter()
    return __default_docling_converter


def convert_to_markdown(document_uri: str):
    """Convert a document to a list of per-page Markdown strings using Docling.

    Parameters
    ----------
    document_uri : str
        The URI or file path of the document to convert.

    Returns
    -------
    List[str]
        One Markdown string per page. If the document has no page
        structure, a single-element list is returned.

    Raises
    ------
    ImportError
        If docling is not installed.
    """
    # Check if docling is available, if not throw an error!
    if not is_docling_available():
        raise ImportError(
            "docling is not available. Please install it to use this function."
        )

    # Convert the document URI to a local file path.
    document_path = uri_to_local_file_path(document_uri)

    # Get the converter (lazy initialization)
    converter = _get_docling_converter()

    # Parse the document:
    result = converter.convert(document_path)

    # Get the document:
    document = result.document

    # List to hold the text of each page:
    pages_text = []

    # Check if the document has pages:
    if document.pages:

        # Extract text from each page of the document:
        for i, page in enumerate(document.pages):
            # Extract text for this page
            page_text = page.export_to_markdown()

            # Append the text to the list:
            pages_text.append(page_text)

    else:
        # If the document does not have pages, we can still extract text from the document:
        page_text = document.export_to_markdown()
        pages_text.append(page_text)

    # Return the text:
    return pages_text
