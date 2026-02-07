import io
import os
from functools import lru_cache
from tempfile import mkdtemp
from typing import List, Tuple

from arbol import aprint
from PIL import Image

from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


@lru_cache()
def is_pymupdf_available() -> bool:
    """
    Check whether the pymupdf and pymupdf4llm packages are available.

    Returns
    -------
    bool
        True if both pymupdf and pymupdf4llm are importable.
    """
    try:
        import importlib.util

        return (
            importlib.util.find_spec("pymupdf4llm") is not None
            and importlib.util.find_spec("pymupdf") is not None
        )

    except ImportError:
        return False


def initialize_docling_converter():
    """
    Initialize the Docling document converter.

    Configures PDF and DOCX pipeline options including OCR, table
    extraction, and page image generation.

    Returns
    -------
    DocumentConverter
        A configured Docling DocumentConverter instance.
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


def convert_document_to_markdown(document_uri: str) -> str:
    """
    Convert a document to a markdown string using pymupdf4llm.

    Parameters
    ----------
    document_uri : str
        The URI of the document to convert. Can be a local path, file URI,
        or remote URL.

    Returns
    -------
    str
        The document content as a markdown-formatted string.

    Raises
    ------
    ImportError
        If pymupdf or pymupdf4llm is not installed.
    """

    # Check if package pymupdf is available:
    if not is_pymupdf_available():
        raise ImportError(
            "pymupdf is not available. Please install it to use this function."
        )

    # Import the necessary modules:
    import pymupdf4llm

    # Download if needed document to temp folder:
    document_path = uri_to_local_file_path(document_uri)

    # Convert the document to markdown:
    md_text = pymupdf4llm.to_markdown(document_path)

    return md_text


def extract_images_from_document(document_uri: str) -> List[str]:
    """
    Extract embedded images from a document using PyMuPDF.

    Each image found in the document pages is saved as a PNG file in a
    temporary directory.

    Parameters
    ----------
    document_uri : str
        The URI of the document to extract images from. Can be a local
        path, file URI, or remote URL.

    Returns
    -------
    List[str]
        A list of ``file://`` URIs pointing to the extracted PNG images.

    Raises
    ------
    ImportError
        If pymupdf is not installed.
    """

    # Check if package pymupdf is available:
    if not is_pymupdf_available():
        raise ImportError(
            "pymupdf is not available. Please install it to use this function."
        )

    # Import the necessary modules:
    import pymupdf

    # We need a temporary folder obtained with tempfile and mkdtemp to store the images:
    temp_image_directory_path = mkdtemp(prefix="extracted_images_")

    # Download if needed document to temp folder:
    document_path = uri_to_local_file_path(document_uri)

    # Extract images from the document:
    doc = pymupdf.open(document_path)  # open a document

    # List of files to return:
    image_files_uris = []

    for page_index in range(len(doc)):  # iterate over pdf pages
        page = doc[page_index]  # get the page
        image_list = page.get_images()

        # print the number of images found on the page
        if image_list:
            aprint(f"Found {len(image_list)} images on page {page_index}")
        else:
            aprint("No images found on page", page_index)

        for image_index, img in enumerate(
            image_list, start=1
        ):  # enumerate the image list
            xref = img[0]  # get the XREF of the image
            pix = pymupdf.Pixmap(doc, xref)  # create a Pixmap

            if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            # Image file name:
            image_file_name = f"page_{page_index}-image_{image_index}.png"

            # Absolute path of the image file in temp directory:
            image_file_path = os.path.join(temp_image_directory_path, image_file_name)

            # Save the image as a PNG file:
            pix.save(image_file_path)

            # Close the pixmap:
            del pix

            # Append the image file name to the list:
            image_files_uris.append("file://" + image_file_path)

    # Close the document:
    doc.close()

    # Return the list of image file uris:
    return image_files_uris


def take_images_of_each_document_page(document_uri, dpi=300):
    """
    Render each page of a PDF document as an image using PyMuPDF.

    Parameters
    ----------
    document_uri : str
        The URI of the PDF document. Can be a local path, file URI,
        or remote URL.
    dpi : int, optional
        Resolution for the output images in dots per inch. Default is 300.
        The zoom factor is computed as ``dpi / 72``.

    Returns
    -------
    List[str]
        A list of ``file://`` URIs pointing to the rendered PNG images,
        one per page.

    Raises
    ------
    ImportError
        If pymupdf is not installed.
    """

    import fitz  # PyMuPDF

    # Check if package pymupdf is available:
    if not is_pymupdf_available():
        raise ImportError(
            "pymupdf is not available. Please install it to use this function."
        )

    # Convert the URI to a local file path:
    document_path = uri_to_local_file_path(document_uri)

    # We need a temporary folder obtained with tempfile and mkdtemp to store the images:
    temp_image_directory_path = mkdtemp(prefix="extracted_page_images_")

    # Calculate zoom factor (default PDF DPI is 72)
    zoom = dpi / 72

    # Create a matrix for the zoom factor
    mat = fitz.Matrix(zoom, zoom)

    # Open the document using PyMuPDF:
    doc = fitz.open(document_path)

    # List to store the image URIs:
    image_uris = []

    for page_index, page in enumerate(doc):
        # Render the page to an image using the zoom matrix:
        pix = page.get_pixmap(matrix=mat)

        # Image file name:
        image_file_name = f"image_for_page_{page_index}.png"

        # Absolute path of the image file in temp directory:
        image_file_path = os.path.join(temp_image_directory_path, image_file_name)

        # Save the image as a PNG file:
        pix.save(image_file_path)

        # Append the image file name to the list:
        image_uris.append("file://" + image_file_path)

        # Close the pixmap:
        del pix

    # Close the document:
    doc.close()

    return image_uris


def extract_text_from_document_pages(document_uri: str) -> List[str]:
    """
    Extract text from each page of a document.

    For PDF files, uses PyMuPDF directly. For other formats, uses the
    Docling document converter.

    Parameters
    ----------
    document_uri : str
        The URI of the document to extract text from. Can be a local
        path, file URI, or remote URL.

    Returns
    -------
    List[str]
        A list of text strings, one per page. If the converter does not
        support page-by-page extraction, a single-element list is returned.

    Raises
    ------
    ImportError
        If pymupdf is not installed (for PDF files).
    """

    # Convert the URI to a local file path:
    document_path = uri_to_local_file_path(document_uri)
    # print(f"Extracting text from uri: {document_uri}")
    # print(f"Extracting text from path: {document_path}")

    # List to store the text of each page:
    pages_text = []

    # If extension is PDF, then we use PyMuPDF:
    if document_uri.endswith(".pdf"):
        import fitz  # PyMuPDF

        # Check if package pymupdf is available:
        if not is_pymupdf_available():
            raise ImportError(
                "pymupdf is not available. Please install it to use this function."
            )

        # Open the document using PyMuPDF:
        doc = fitz.open(document_path)

        # Iterate over each page and extract text:
        for page in doc:
            page_text = page.get_text(
                "text"
            )  # or simply page.get_text() if "text" is default
            pages_text.append(page_text)

        # Close the document:
        doc.close()
    else:
        # Instantiate the DocumentConverter:
        converter = initialize_docling_converter()

        # Parse the document:
        result = converter.convert(document_path)

        # Get the document:
        document = result.document

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

    return pages_text


def extract_text_and_image_from_document(
    document_uri: str, dpi: int = 300
) -> List[Tuple[str, Image.Image]]:
    """
    Extract text and generate an image for each page in the document.

    Parameters
    ----------
    document_uri : str
        The URI or file path of the document to process.
    dpi : int, optional
        The resolution for the output images in dots per inch. Default is 300.

    Returns
    -------
    List[Tuple[str, Image.Image]]
        A list of tuples, each containing:
            - The extracted text from the page (str).
            - A PIL Image object representing the page.
    """

    if document_uri.endswith(".pdf"):
        import fitz  # PyMuPDF

        # Check if package pymupdf is available:
        if not is_pymupdf_available():
            raise ImportError(
                "pymupdf is not available. Please install it to use this function."
            )

        # Convert the document URI to a local file path.
        document_path = uri_to_local_file_path(document_uri)

        try:
            doc = fitz.open(document_path)
        except Exception as e:
            raise RuntimeError(f"Failed to open document '{document_path}': {e}")

        pages_content = []
        # Compute the zoom factor based on the desired DPI (PDFs default to 72 DPI)
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        for page_number in range(len(doc)):
            try:
                # Load the page and extract text
                page = doc.load_page(page_number)
                page_text = page.get_text("text")
                # Render the page to an image using the zoom matrix
                pix = page.get_pixmap(matrix=mat)
                # Convert the pixmap to PNG bytes and then to a PIL Image
                img_bytes = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_bytes))
                pages_content.append((page_text, image))
            except Exception as e:
                aprint(f"Error processing page {page_number}: {e}")
                continue

        doc.close()

    else:
        raise NotImplementedError("Only PDF documents are supported for now.")

    return pages_content
