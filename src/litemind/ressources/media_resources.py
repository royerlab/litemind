class MediaResources:
    """
    A class that provides methods to get the local test file URIs for images, audio, videos, documents, tables, and archives.
    This class is used for testing purposes only.
    It is not intended to be used in production code.

    See files in the `ressources` folder for the different test files.
    """

    @staticmethod
    def get_local_test_folder_path(folder_name: str):
        import os

        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)

        # Combine the two to get the absolute path
        absolute_path = os.path.join(current_dir, folder_name)

        return absolute_path

    @staticmethod
    def get_local_test_file_uri(filetype, image_name):
        import os

        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)
        # Combine the two to get the absolute path
        absolute_path = os.path.join(
            current_dir, os.path.join(f"{filetype}/", image_name)
        )
        uri = "file://" + absolute_path
        # print(uri)
        return uri

    @staticmethod
    def get_local_test_image_uri(image_name: str):
        return MediaResources.get_local_test_file_uri("images", image_name)

    @staticmethod
    def get_local_test_ndimage_uri(image_name: str):
        return MediaResources.get_local_test_file_uri("ndimages", image_name)

    @staticmethod
    def get_local_test_audio_uri(image_name: str):
        return MediaResources.get_local_test_file_uri("audio", image_name)

    @staticmethod
    def get_local_test_video_uri(image_name: str):
        return MediaResources.get_local_test_file_uri("videos", image_name)

    @staticmethod
    def get_local_test_document_uri(doc_name: str):
        return MediaResources.get_local_test_file_uri("documents", doc_name)

    @staticmethod
    def get_local_test_table_uri(doc_name: str):
        return MediaResources.get_local_test_file_uri("tables", doc_name)

    @staticmethod
    def get_local_test_archive_uri(doc_name: str):
        return MediaResources.get_local_test_file_uri("archives", doc_name)

    @staticmethod
    def get_local_test_other_uri(doc_name: str):
        return MediaResources.get_local_test_file_uri("others", doc_name)
