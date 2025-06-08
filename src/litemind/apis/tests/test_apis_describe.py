import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.apis.base_api import ModelFeatures
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_document import Document
from litemind.media.types.media_image import Image
from litemind.media.types.media_video import Video
from litemind.ressources.media_resources import MediaResources


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsDescribe(MediaResources):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    These tests are for the describe methods of the API.
    """

    def test_describe_image_if_supported(self, api_class):
        """
        Test describe_image if the implementation supports images.
        If not, we skip it.
        """
        api_instance = api_class()

        # Get the best model for the requested features:
        default_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration],
            media_types=[Image],
        )

        # If no model is found, we skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support images. Skipping image tests."
            )

        try:

            # Get the path to the test image:
            image_path = self.get_local_test_image_uri("future.jpeg")

            # Call the describe_image method:
            description = api_instance.describe_image(
                image_path, model_name=default_model_name
            )

            print("\n" + description)

            assert isinstance(
                description, str
            ), f"{api_class.__name__}.describe_image() should return a string!"
            assert (
                len(description) > 0
            ), f"{api_class.__name__}.describe_image() should return a non-empty string!"

            # Check the content of the string:
            assert (
                "robot" in description
                or "futuristic" in description
                or "sky" in description
            )

        except:
            # Print stacktrace:
            import traceback

            traceback.print_exc()

            # If exception happened then tests failed:
            assert False

    def test_describe_audio_if_supported(self, api_class):
        """
        Test describe_audio if the implementation supports audio.
        If not, we skip it.
        """
        api_instance = api_class()

        # Get the best model for the requested features:
        default_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration],
            media_types=[Audio],
        )

        # If no model is found, we skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support audio. Skipping audio tests."
            )

        try:
            # Get the path to the test audio:
            audio_path = self.get_local_test_audio_uri("harvard.wav")

            # Describe the audio:
            description = api_instance.describe_audio(
                audio_path, model_name=default_model_name
            )

            print("\n" + description)

            assert isinstance(
                description, str
            ), f"{api_class.__name__}.describe_audio() should return a string!"
            assert (
                len(description) > 0
            ), f"{api_class.__name__}.describe_audio() should return a non-empty string!"

            # Lower case:
            description = description.lower()

            # Check the contents of the string:
            assert (
                "smell" in description
                or "ham" in description
                or "beer" in description
                or "sentence" in description
            )

        except:
            # Print stacktrace:
            import traceback

            traceback.print_exc()

            # If exception happened then tests failed:
            assert False

    def test_describe_video_if_supported(self, api_class):
        """
        Test describe_video if the implementation supports videos.
        If not, we skip it.
        """
        api_instance = api_class()

        # Get the best model for the requested features:
        default_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration],
            media_types=[Video],
        )

        # If no model is found, we skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support videos. Skipping video tests."
            )

        try:
            # Get the path to the test video:
            video_path = self.get_local_test_video_uri("lunar_park.mov")

            # Describe the video:
            description = api_instance.describe_video(
                video_path, model_name=default_model_name
            )

            print("\n" + description)

            assert isinstance(
                description, str
            ), f"{api_class.__name__}.describe_video() should return a string!"
            assert (
                len(description) > 0
            ), f"{api_class.__name__}.describe_video() should return a non-empty string!"

            # Lower case:
            description = description.lower()

            # If ApiClass.__name__ is 'OllamaApi', we relax the tests to allow for more flexibility:
            if api_class.__name__ == "OllamaApi":
                # Open source models are not yet strong enough to understand that a sequence of images is a video:
                assert "elephant" in description or "people" in description
            else:

                # Check the contents of the string:
                assert (
                    "roller coaster" in description
                    or "amusement park" in description
                    or "ride" in description
                ) and (
                    "20th century" in description
                    or "20th-century" in description
                    or "earlier era" in description
                    or "vintage" in description
                    or "period" in description
                )

        except:
            # Print stacktrace:
            import traceback

            traceback.print_exc()

            # If exception happened then tests failed:
            assert False

    def test_describe_document_if_supported(self, api_class):
        """
        Test describe_document if the implementation supports documents.
        If not, we skip it.
        """
        api_instance = api_class()

        # Get the best model for the requested features:
        default_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration],
            media_types=[Document],
        )

        # If no model is found, we skip the test:
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support documents. Skipping documents tests."
            )

        try:
            # Get the path to the test video:
            document_path = self.get_local_test_document_uri("noise2self_paper.pdf")

            # Describe the video:
            description = api_instance.describe_document(
                document_path, model_name=default_model_name
            )

            print("\n" + description)

            assert isinstance(
                description, str
            ), f"{api_class.__name__}.describe_document() should return a string!"
            assert (
                len(description) > 0
            ), f"{api_class.__name__}.describe_document() should return a non-empty string!"

            # Lower case:
            description = description.lower()

            # If ApiClass.__name__ is 'OllamaApi', we relax the tests to allow for more flexibility:
            if api_class.__name__ == "OllamaApi":
                # Open source models are not yet strong enough to understand that a sequence of images is a video:
                assert "document" in description or "citations" in description
            else:

                # Check the contents of the string:
                assert (
                    "j-invariance" in description
                    or "noise" in description
                    or "parameters" in description
                )

        except:
            # Print stacktrace:
            import traceback

            traceback.print_exc()

            # If exception happened then tests failed:
            assert False
