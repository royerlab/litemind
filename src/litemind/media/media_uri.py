import os
from abc import abstractmethod
from typing import Optional

from litemind.media.media_default import MediaDefault
from litemind.utils.get_media_type_from_uri import get_media_type_from_uri
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path
from litemind.utils.read_file_and_convert_to_base64 import (
    base64_to_data_uri,
    read_file_and_convert_to_base64,
)
from litemind.utils.uri_utils import is_uri, is_valid_path


class MediaURI(MediaDefault):
    """
    A media that is accessible via URI.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """
        Create a new URI-based media.

        Parameters
        ----------
        uri: str
            The media's URI (Uniform Resource Identifier) or local file path.
        extension: str
            Extension/Type of the media file in case it is not clear from the URI.
        kwargs: dict
            Other arguments passed to MediaDefault.

        """
        super().__init__(**kwargs)

        # Check that it is a valid table URI:
        if not is_uri(uri) or not is_valid_path(uri):
            raise ValueError(f"Invalid URI or local file path: '{uri}'.")

        # Check that file is a valid file path and normalise to URI:
        if not is_uri(uri):
            # Get the absolute file path and then prepend 'file://:
            uri = os.path.abspath(uri)
            uri = "file://" + uri

        # Set attributes:
        self.uri = uri
        self.extension = extension.lower() if extension else None
        self.local_path = None

    def get_content(self) -> str:
        """
        Get the content of the media.

        Returns
        -------
        str
            The content of the media.
        """
        return self.uri

    def is_local(self):
        """
        Check if the URI is local.
        """
        return self.uri.startswith("file://") or self.uri.startswith("/")

    def get_filename(self):
        """
        Get the filename of the media.
        """
        return self.uri.split("/")[-1] if self.uri else None

    def get_extension(self):
        """
        Get the extension of the media, without the dot
        """
        return self.extension or self.uri.split(".")[-1].lower()

    def has_extension(self, extension: str):
        """
        Check if the media has an extension.

        Parameters
        ----------
        extension: str

        Returns
        -------
        bool

        """

        # Normalize the extension to lower case:
        extension = extension.lower()

        # Check if the extension is in the media's extension or if the URI ends with the extension:
        return extension.lower() in self.get_extension() or self.uri.endswith(extension)

    def get_media_type(self) -> Optional[str]:
        """
        Get the media type of this media.
        Returns
        -------
        str
            The media type of this media

        """
        return get_media_type_from_uri(self.uri)

    def get_mime_type(self, mime_prefix) -> Optional[str]:
        """
        Get the MIME type of this media.

        Returns
        -------
        str
            The MIME type of this media.

        """
        return f"{mime_prefix}/{self.get_extension()}"

    def to_remote_or_data_uri(self):
        """
        Convert the URI to a remote or data URI.

        Returns
        -------
        str
            The remote or data URI.

        """

        # Default value is URI:
        uri = self.uri

        if uri.startswith("file://"):
            # If local file URI then we make it into a data URI:
            media_type = get_media_type_from_uri(uri)
            local_path = uri.replace("file://", "")
            base64_data = read_file_and_convert_to_base64(local_path)
            uri = base64_to_data_uri(base64_data, media_type)

        return uri

    def to_base64_data(self):
        """
        Convert the URI to a base64 data (just the data part of the URI).
        Returns
        -------
        str
            The base64 data of the URI.
        """
        # If the URI is a data URI, then we return the data part:
        if self.uri.startswith("data:"):
            return self.uri.split(",")[-1]

        # If not, first we normalise the URI to  a local file path:
        local_path = uri_to_local_file_path(self.uri)

        # Then we read the file and convert it to base64:
        base64_data = read_file_and_convert_to_base64(local_path)

        return base64_data

    def to_local_file_path(self) -> str:
        """
        Convert the URI of this media to a local file path.

        Parameters
        ----------

        Returns
        -------
        str
            The local file path of the URI.
        """

        if self.local_path:
            # If the local path is already set, return it:
            return self.local_path

        # Normalise the URI to a local file path:
        self.local_path = uri_to_local_file_path(self.uri)

        return self.local_path

    @abstractmethod
    def load_from_uri(self):
        """
        Load the data from the URI.
        """
        pass

    def __str__(self):
        return self.uri

    def __repr__(self):
        return f"MediaURI({self.uri})"

    def __len__(self):
        return len(self.uri)
