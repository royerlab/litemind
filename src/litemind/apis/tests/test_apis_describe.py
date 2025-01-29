import pytest

from litemind.apis.anthropic.anthropic_api import AnthropicApi
from litemind.apis.base_api import ModelFeatures
from litemind.apis.google.google_api import GeminiApi
from litemind.apis.ollama.ollama_api import OllamaApi
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.apis.tests.base_test import BaseTest

# Put all your implementations in this list:
API_IMPLEMENTATIONS = [
    OpenAIApi,
    OllamaApi,
    AnthropicApi,
    GeminiApi
]


@pytest.mark.parametrize("ApiClass", API_IMPLEMENTATIONS)
class TestBaseApiImplementations(BaseTest):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    """

    def test_describe_image_if_supported(self, ApiClass):
        """
        Test describe_image if the implementation supports images.
        If not, we skip it.
        """
        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])

        if not api_instance.has_model_support_for(model_name=default_model_name,
                                                  features=[
                                                      ModelFeatures.TextGeneration,
                                                      ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image tests.")

        try:

            image_path = self._get_local_test_image_uri('future.jpeg')

            description = api_instance.describe_image(image_path,
                                                      model_name=default_model_name)

            print('\n' + description)

            assert isinstance(description, str), (
                f"{ApiClass.__name__}.describe_image() should return a string!"
            )
            assert len(description) > 0, (
                f"{ApiClass.__name__}.describe_image() should return a non-empty string!"
            )

            # Check the content of the string:
            assert 'robot' in description or 'futuristic' in description or 'sky' in description

        except:
            # Print stacktrace:
            import traceback
            traceback.print_exc()

            # If exception happened then tests failed:
            assert False

    def test_describe_audio_if_supported(self, ApiClass):
        """
        Test describe_audio if the implementation supports audio.
        If not, we skip it.
        """
        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Audio])

        if not api_instance.has_model_support_for(model_name=default_model_name,
                                                  features=[
                                                      ModelFeatures.TextGeneration,
                                                      ModelFeatures.Audio]):
            pytest.skip(
                f"{ApiClass.__name__} does not support audio. Skipping audio tests.")

        try:
            audio_path = self._get_local_test_audio_uri('harvard.wav')

            description = api_instance.describe_audio(audio_path,
                                                      model_name=default_model_name)

            print('\n' + description)

            assert isinstance(description, str), (
                f"{ApiClass.__name__}.describe_audio() should return a string!"
            )
            assert len(description) > 0, (
                f"{ApiClass.__name__}.describe_audio() should return a non-empty string!"
            )

            # Check the contents of the string:
            assert 'smell' in description or 'ham' in description or 'beer' in description

        except:
            # Print stacktrace:
            import traceback
            traceback.print_exc()

            # If exception happened then tests failed:
            assert False

    def test_describe_video_if_supported(self, ApiClass):
        """
        Test describe_video if the implementation supports videos.
        If not, we skip it.
        """
        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Video])

        if not api_instance.has_model_support_for(model_name=default_model_name,
                                                  features=[
                                                      ModelFeatures.TextGeneration,
                                                      ModelFeatures.Video]):
            pytest.skip(
                f"{ApiClass.__name__} does not support videos. Skipping video tests.")

        try:
            video_path = self._get_local_test_video_uri('lunar_park.mov')

            description = api_instance.describe_video(video_path,
                                                      model_name=default_model_name)

            print('\n' + description)

            assert isinstance(description, str), (
                f"{ApiClass.__name__}.describe_video() should return a string!"
            )
            assert len(description) > 0, (
                f"{ApiClass.__name__}.describe_video() should return a non-empty string!"
            )

            # Lower case:
            description = description.lower()

            # If ApiClass.__name__ is 'OllamaApi', we relax the tests to allow for more flexibility:
            if ApiClass.__name__ == 'OllamaApi':
                # Open source models are not yet strong enough to understand that a sequence of images is a video:
                assert 'elephant' in description or 'people' in description
            else:

                # Check the contents of the string:
                assert ((
                                'roller coaster' in description or 'amusement park' in description)
                        and (
                                '20th century' in description or '20th-century' in description))

        except:
            # Print stacktrace:
            import traceback
            traceback.print_exc()

            # If exception happened then tests failed:
            assert False
