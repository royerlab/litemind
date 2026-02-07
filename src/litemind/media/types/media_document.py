from typing import List, Optional

from litemind.media.media_uri import MediaURI
from litemind.utils.document_processing import (
    extract_text_from_document_pages,
    take_images_of_each_document_page,
)
from litemind.utils.file_types.file_types import is_document_file
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class Document(MediaURI):
    """Media that stores a multipage document such as a PDF.

    Supports PDF, DOCX, PPTX, and other document formats. Provides methods
    to extract text, render page images, and convert to Markdown.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """Create a new document media.

        Parameters
        ----------
        uri : str
            The document URI or local file path. Remote HTTP(S) URLs are
            accepted without extension validation.
        extension : str, optional
            File extension override without the leading dot.
        **kwargs
            Additional keyword arguments forwarded to ``MediaURI``.

        Raises
        ------
        ValueError
            If the URI is a local file and does not have a recognised
            document file extension.
        """

        if uri.startswith("http"):
            pass
        elif uri.startswith("file://"):  # Local file URI
            file_path = uri[7:]

            if not is_document_file(file_path):
                # If it is a remote URL, then it is ok to not see a correct extension!
                raise ValueError(
                    f"Invalid local document URI: '{uri}' (must have a valid document file extension)"
                )

        super().__init__(uri=uri, extension=extension, **kwargs)

        self.markdown = None

    def load_from_uri(self):
        """Load the document by converting it to Markdown.

        Populates ``self.markdown`` with the Markdown representation.
        """

        # load the document:
        self.markdown = self.to_markdown()

    def extract_text_from_pages(self) -> List[str]:
        """Extract the text content of each page.

        Returns
        -------
        List[str]
            A list of strings, one per page, containing the extracted text.
        """

        # Download the document file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # Extract text from the document pages:
        pages = extract_text_from_document_pages(local_file)

        return pages

    def take_image_of_each_page(self):
        """Render each document page as an image.

        Returns
        -------
        list
            A list of image representations, one per page.
        """

        # Download the document file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # Create images of each page:
        images = take_images_of_each_document_page(local_file)

        return images

    def to_markdown(self, media_converter: Optional["MediaConverter"] = None):
        """Convert the document content to a Markdown string.

        Parameters
        ----------
        media_converter : MediaConverter, optional
            An optional media converter to use during the conversion.

        Returns
        -------
        str
            The Markdown representation of the document.
        """

        # create a Message from the local document file:
        from litemind.agent.messages.message import Message

        message = Message(role="user")
        message.append_document(document_uri=self.uri)

        # Convert the document to markdown:
        markdown_string = message.to_markdown(media_converter=media_converter)

        return markdown_string
