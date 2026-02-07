"""
Test media resource locator.

Provides ``MediaResources``, a helper class that resolves ``file://`` URIs
for the sample media files bundled in the ``ressources`` directory. These
files are used exclusively in the test suite.
"""


class MediaResources:
    """
    Locate bundled sample media files for testing.

    Every static method resolves to a ``file://`` URI pointing at a test
    asset stored alongside this module in the ``ressources`` sub-directories
    (``images/``, ``audio/``, ``videos/``, etc.).

    This class is intended for the test suite only and should not be used
    in production code.
    """

    @staticmethod
    def get_local_test_folder_path(folder_name: str) -> str:
        """
        Return the absolute filesystem path of a resource sub-folder.

        Parameters
        ----------
        folder_name : str
            Name of the sub-folder inside the ``ressources`` directory
            (e.g. ``"images"``, ``"audio"``).

        Returns
        -------
        str
            Absolute path to the requested folder.
        """
        import os

        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)

        # Combine the two to get the absolute path
        absolute_path = os.path.join(current_dir, folder_name)

        return absolute_path

    @staticmethod
    def get_local_test_file_uri(filetype: str, file_name: str) -> str:
        """
        Return a ``file://`` URI for a test resource file.

        Parameters
        ----------
        filetype : str
            The resource sub-directory name (e.g. ``"images"``,
            ``"audio"``, ``"videos"``).
        file_name : str
            Name of the file within the *filetype* sub-directory.

        Returns
        -------
        str
            A ``file://`` URI pointing at the resource.
        """
        import os

        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)
        # Combine the two to get the absolute path
        absolute_path = os.path.join(
            current_dir, os.path.join(f"{filetype}/", file_name)
        )
        uri = "file://" + absolute_path
        return uri

    @staticmethod
    def get_local_test_image_uri(image_name: str) -> str:
        """Return a ``file://`` URI for a test image."""
        return MediaResources.get_local_test_file_uri("images", image_name)

    @staticmethod
    def get_local_test_ndimage_uri(image_name: str) -> str:
        """Return a ``file://`` URI for a test n-dimensional image."""
        return MediaResources.get_local_test_file_uri("ndimages", image_name)

    @staticmethod
    def get_local_test_audio_uri(audio_name: str) -> str:
        """Return a ``file://`` URI for a test audio file."""
        return MediaResources.get_local_test_file_uri("audio", audio_name)

    @staticmethod
    def get_local_test_video_uri(video_name: str) -> str:
        """Return a ``file://`` URI for a test video file."""
        return MediaResources.get_local_test_file_uri("videos", video_name)

    @staticmethod
    def get_local_test_document_uri(doc_name: str) -> str:
        """Return a ``file://`` URI for a test document."""
        return MediaResources.get_local_test_file_uri("documents", doc_name)

    @staticmethod
    def get_local_test_table_uri(table_name: str) -> str:
        """Return a ``file://`` URI for a test table file."""
        return MediaResources.get_local_test_file_uri("tables", table_name)

    @staticmethod
    def get_local_test_archive_uri(archive_name: str) -> str:
        """Return a ``file://`` URI for a test archive file."""
        return MediaResources.get_local_test_file_uri("archives", archive_name)

    @staticmethod
    def get_local_test_other_uri(file_name: str) -> str:
        """Return a ``file://`` URI for a miscellaneous test file."""
        return MediaResources.get_local_test_file_uri("others", file_name)
