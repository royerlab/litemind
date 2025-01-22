import os
from tempfile import mkdtemp
from typing import List

from arbol import aprint

from litemind.utils.normalise_uri_to_local_file_path import \
    uri_to_local_file_path


def is_pymupdf_available() -> bool:
    # Check if package pymupdf s available:
    try:
        import pymupdf
        import pymupdf4llm
        return True
    except ImportError:
        return False


def convert_document_to_markdown(document_uri: str) -> str:
    """
    Convert a document to a markdown string.
    
    Parameters
    ----------
    document_uri: str
        The URI of the document to convert.

    Returns
    -------
    str
        The resulting markdown string.

    """

    # Check if package pymupdf is available:
    if not is_pymupdf_available():
        raise ImportError(
            "pymupdf is not available. Please install it to use this function.")

    # Import the necessary modules:
    import pymupdf4llm

    # Download if needed document to temp folder:
    document_path = uri_to_local_file_path(document_uri)

    # Convert the document to markdown:
    md_text = pymupdf4llm.to_markdown(document_path)

    return md_text


def extract_images_from_document(document_uri: str) -> List[str]:
    """
    Extract images from a document.
    
    Parameters
    ----------
    document_uri: str
        The URI of the document to extract images from.
        
    Returns
    -------
    List[str]
        The list of image file URIs.
        
    """

    # Check if package pymupdf is available:
    if not is_pymupdf_available():
        raise ImportError(
            "pymupdf is not available. Please install it to use this function.")

    # Import the necessary modules:
    import pymupdf

    # We need a temporary folder obtained with tempfile and mkdtemp to store the images:
    temp_image_directory_path = mkdtemp(prefix='extracted_images_')

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

        for image_index, img in enumerate(image_list,
                                          start=1):  # enumerate the image list
            xref = img[0]  # get the XREF of the image
            pix = pymupdf.Pixmap(doc, xref)  # create a Pixmap

            if pix.n - pix.alpha > 3:  # CMYK: convert to RGB first
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)

            # Image file name:
            image_file_name = f"page_{page_index}-image_{image_index}.png"

            # Absolute path of the image file in temp directory:
            image_file_path = os.path.join(temp_image_directory_path,
                                           image_file_name)

            # Save the image as a PNG file:
            pix.save(image_file_path)

            # Close the pixmap:
            pix = None

            # Append the image file name to the list:
            image_files_uris.append('file://' + image_file_path)

    # Close the document:
    doc.close()

    # Return the list of image file uris:
    return image_files_uris
