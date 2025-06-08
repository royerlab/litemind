from typing import List, Optional

from litemind.media.media_uri import MediaURI
from litemind.utils.document_processing import (
    extract_text_from_document_pages,
    take_images_of_each_document_page,
)
from litemind.utils.file_types.file_types import is_document_file
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class Document(MediaURI):
    """
    A media document that stores a multipage document such as a PDF.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):

        if uri.startswith("http"):
            pass
        elif uri.startswith("local:"):  # Local file URI
            file_path = uri[6:]

            if not is_document_file(file_path):
                # If it is a remote URL, then it is ok to not see a correct extension!
                raise ValueError(
                    f"Invalid local document URI: '{uri}' (must have a valid document file extension)"
                )

        super().__init__(uri=uri, extension=extension, **kwargs)

        self.markdown = None

    def load_from_uri(self):

        # load the document:
        self.markdown = self.to_markdown()

    def extract_text_from_pages(self) -> List[str]:

        # Download the video file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # Extract text from the document pages:
        pages = extract_text_from_document_pages(local_file)

        return pages

    def take_image_of_each_page(self):

        # Download the video file from the URI to a local file:
        local_file = uri_to_local_file_path(self.uri)

        # Create images of each page:
        images = take_images_of_each_document_page(local_file)

        return images

    def to_markdown(self, media_converter: Optional["MediaConverter"] = None):

        # create a Message from the local document file:
        from litemind.agent.messages.message import Message

        message = Message(role="user")
        message.append_document(document_uri=self.uri)

        # Convert the document to markdown:
        markdown_string = message.to_markdown(media_converter=media_converter)

        return markdown_string
