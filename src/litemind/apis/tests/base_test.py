from litemind.apis.anthropic.anthropic_api import AnthropicApi
from litemind.apis.google.google_api import GeminiApi
from litemind.apis.ollama.ollama_api import OllamaApi
from litemind.apis.openai.openai_api import OpenAIApi

# Put all your implementations in this list:
API_IMPLEMENTATIONS = [
    OpenAIApi,
    OllamaApi,
    AnthropicApi,
    GeminiApi
]


class BaseTest:

    def _get_local_test_image_uri(self, image_name: str):
        return self._get_local_test_file_uri('images', image_name)

    def _get_local_test_audio_uri(self, image_name: str):
        return self._get_local_test_file_uri('audio', image_name)

    def _get_local_test_video_uri(self, image_name: str):
        return self._get_local_test_file_uri('videos', image_name)

    def _get_local_test_document_uri(self, doc_name: str):
        return self._get_local_test_file_uri('documents', doc_name)

    def _get_local_test_table_uri(self, doc_name: str):
        return self._get_local_test_file_uri('tables', doc_name)

    def _get_local_test_archive_uri(self, doc_name: str):
        return self._get_local_test_file_uri('archives', doc_name)

    def _get_local_test_folder_path(self, folder_name: str):
        import os
        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)

        # Combine the two to get the absolute path
        absolute_path = os.path.join(current_dir, folder_name)

        return absolute_path

    def _get_local_test_file_uri(self, filetype, image_name):
        import os
        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)
        # Combine the two to get the absolute path
        absolute_path = os.path.join(current_dir,
                                     os.path.join(f'{filetype}/', image_name))
        uri = 'file://' + absolute_path
        print(uri)
        return uri
