from typing import List, Optional

from litemind.media.media_uri import MediaURI
from litemind.utils.document_processing import (
    extract_text_and_image_from_document,
    extract_text_from_document_pages,
    take_images_of_each_document_page,
)
from litemind.utils.file_extensions import document_file_extensions
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class Document(MediaURI):
    """
    A media document that stores a multipage document such as a PDF.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):

        # Check that the file extension is valid:
        if not any(uri.endswith(ext) for ext in document_file_extensions):
            # If it is a remote URL, then it is ok to not see a correct extension!
            if uri.startswith("http"):
                pass
            else:
                raise ValueError(
                    f"Invalid document URI: '{uri}' (must have a valid document file extension)"
                )

        super().__init__(uri=uri, extension=extension, **kwargs)

    def load_from_uri(self):

        # Download the video file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # load the document:
        self.pages = extract_text_and_image_from_document(local_file)

    def extract_text_from_pages(self) -> List[str]:

        # Extract text from the document pages:
        pages = extract_text_from_document_pages(self.uri)

        return pages

    def take_image_of_each_page(self):

        # Create images of each page:
        images = take_images_of_each_document_page(self.uri)

        return images
