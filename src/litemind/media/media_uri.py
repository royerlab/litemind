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
    """Media whose content is referenced by a URI.

    Supports ``file://`` URIs for local files and ``http(s)://`` URIs for
    remote resources. Local file paths are automatically converted to
    ``file://`` URIs on construction. Provides helper methods for resolving
    URIs to local paths, base64 data, and MIME types.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """Create a new URI-based media.

        Parameters
        ----------
        uri : str
            A URI (``file://``, ``http://``, ``https://``) or local file path.
        extension : str, optional
            File extension override (without the leading dot, e.g. ``"png"``).
            Used when the extension cannot be inferred from the URI.
        **kwargs
            Additional keyword arguments forwarded to ``MediaDefault``.

        Raises
        ------
        ValueError
            If *uri* is neither a valid URI nor a valid local file path.
        """
        super().__init__(**kwargs)

        # Check that it is a valid URI or local file path:
        if not uri or (not is_uri(uri) and not is_valid_path(uri)):
            raise ValueError(f"Invalid URI or local file path: '{uri}'.")

        # Check that file is a valid file path and normalise to URI:
        if not is_uri(uri):
            # Get the absolute file path and then prepend 'file://':
            uri = os.path.abspath(uri)
            uri = "file://" + uri

        # Set attributes:
        self.uri = uri
        self.extension = extension.lower() if extension else None
        self.local_path = None

    def get_content(self) -> str:
        """Get the URI string for this media.

        Returns
        -------
        str
            The URI of the media resource.
        """
        return self.uri

    def is_local(self):
        """Check whether the URI points to a local file.

        Returns
        -------
        bool
            True if the URI starts with ``file://`` or ``/``.
        """
        return self.uri.startswith("file://") or self.uri.startswith("/")

    def get_filename(self):
        """Extract the filename from the URI.

        Returns
        -------
        str or None
            The last path component of the URI, or None if the URI is empty.
        """
        return self.uri.split("/")[-1] if self.uri else None

    def get_extension(self):
        """Get the file extension, without the leading dot.

        Returns the explicitly set extension if available, otherwise infers
        it from the URI.

        Returns
        -------
        str
            The file extension in lowercase (e.g. ``"png"``, ``"csv"``).
        """
        if self.extension:
            return self.extension
        # Extract the filename portion from the URI, stripping query/fragment:
        filename = self.uri.split("/")[-1] if "/" in self.uri else self.uri
        filename = filename.split("?")[0].split("#")[0]
        if "." in filename:
            return filename.rsplit(".", 1)[-1].lower()
        return ""

    def has_extension(self, extension: str):
        """Check whether this media has a given file extension.

        Parameters
        ----------
        extension : str
            The extension to check (without the dot, e.g. ``"pdf"``).

        Returns
        -------
        bool
            True if the media's extension matches or the URI ends with the
            given extension.
        """

        # Normalize the extension to lower case:
        extension = extension.lower()

        # Check if the extension is in the media's extension or if the URI ends with the extension:
        return extension.lower() == self.get_extension() or self.uri.lower().endswith(
            "." + extension
        )

    def get_media_type(self) -> Optional[str]:
        """Get the media type string (e.g. ``"image"``, ``"audio"``).

        Returns
        -------
        str or None
            The media type derived from the URI, or None if it cannot be
            determined.
        """
        return get_media_type_from_uri(self.uri)

    def get_mime_type(self, mime_prefix) -> Optional[str]:
        """Build a MIME type string from a prefix and the file extension.

        Parameters
        ----------
        mime_prefix : str
            The MIME type prefix (e.g. ``"image"``, ``"audio"``).

        Returns
        -------
        str
            A MIME type such as ``"image/png"`` or ``"audio/wav"``.
        """
        return f"{mime_prefix}/{self.get_extension()}"

    def to_remote_or_data_uri(self):
        """Convert to a remote URL or an inline ``data:`` URI.

        For local ``file://`` URIs the file is read, base64-encoded, and
        returned as a ``data:`` URI. Remote URIs are returned unchanged.

        Returns
        -------
        str
            A remote URL or a ``data:`` URI.
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
        """Get the raw base64-encoded data for this media.

        For ``data:`` URIs the data portion is extracted directly. For other
        URIs the resource is first resolved to a local file and then encoded.

        Returns
        -------
        str
            The base64-encoded content string (without the ``data:`` prefix).
        """
        # If the URI is a data URI, then we return the data part:
        if self.uri.startswith("data:"):
            return self.uri.split(",")[-1]

        # If not, first we normalise the URI to a local file path:
        local_path = uri_to_local_file_path(self.uri)

        # Then we read the file and convert it to base64:
        base64_data = read_file_and_convert_to_base64(local_path)

        return base64_data

    def to_local_file_path(self) -> str:
        """Resolve the URI to a local file path.

        Remote resources are downloaded to a temporary file on first access.
        The result is cached in ``self.local_path`` for subsequent calls.

        Returns
        -------
        str
            An absolute local file path.
        """

        if self.local_path:
            # If the local path is already set, return it:
            return self.local_path

        # Normalise the URI to a local file path:
        self.local_path = uri_to_local_file_path(self.uri)

        return self.local_path

    @abstractmethod
    def load_from_uri(self):
        """Load and parse the resource data from the URI.

        Subclasses must implement this to populate type-specific attributes
        (e.g., pixel arrays for images, DataFrames for tables).
        """
        pass

    def __str__(self):
        return self.uri

    def __repr__(self):
        return f"MediaURI({self.uri})"

    def __len__(self):
        return len(self.uri)
