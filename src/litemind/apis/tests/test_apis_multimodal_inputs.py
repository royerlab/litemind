import pytest

from litemind.agent.message import Message
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

    def test_text_generation_with_image_url(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image tests.")

        print('\n' + default_model_name)

        image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg'
        print(image_url)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text('Can you describe what you see in the image?')
        user_message.append_image(image_url)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert 'sepia' in response or 'chalkboard' in response or 'Einstein' in response

    def test_text_generation_with_png_image_path(self, ApiClass):
        api_instance = ApiClass()

        # Get the default model name:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])
        print('\n' + default_model_name)

        # Skip tests if the model does not support images
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image tests.")

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text('Can you describe what you see in the image?')
        image_path = self._get_local_test_image_uri('python.png')
        user_message.append_image(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        if ApiClass.__name__ == 'OllamaApi':
            # open source models are typically not yet strong enough.
            assert 'blue' in response or 'logo' in response
        else:
            assert 'snake' in response or 'serpent' in response or 'python' in response

    def test_text_generation_with_jpg_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image tests.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text('Can you describe what you see in the image?')
        image_path = self._get_local_test_image_uri('future.jpeg')
        user_message.append_image(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert 'robot' in response or 'futuristic' in response or 'sky' in response

    def test_text_generation_with_webp_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image tests.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text('Can you describe what you see in the image?')
        image_path = self._get_local_test_image_uri('beach.webp')
        user_message.append_image(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert 'beach' in response or 'palm' in response or 'sunset' in response

    def test_text_generation_with_gif_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image tests.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you describe what you see in the image?')
        image_path = self._get_local_test_image_uri('field.gif')
        user_message.append_image(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert 'field' in response or 'blue' in response or 'stars' in response

    def test_text_generation_with_multiple_images(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image tests.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you compare these two images? What is similar and what is different?')
        cat_image_path = self._get_local_test_image_uri('cat.jpg')
        panda_image_path = self._get_local_test_image_uri('panda.jpg')
        user_message.append_image(cat_image_path)
        user_message.append_image(panda_image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert (
                       'animals' in response or 'characters' in response) and 'cat' in response and 'panda' in response

    def test_text_generation_with_audio_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Audio])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Audio]):
            pytest.skip(
                f"{ApiClass.__name__} does not support audio. Skipping audio tests.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you describe what you heard in the audio file?')
        image_path = self._get_local_test_audio_uri('harvard.wav')
        user_message.append_audio(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert 'smell' in response or 'ham' in response or 'beer' in response

    def test_text_generation_with_audio_url(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Audio])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Audio]):
            pytest.skip(
                f"{ApiClass.__name__} does not support audio. Skipping audio tests.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you describe in detail what i said the following audio file?')
        user_message.append_audio(
            "https://salford.figshare.com/ndownloader/files/14630270")

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert 'canoe' in response or 'chicken' in response or 'hours' in response

    def test_text_generation_with_video_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Video])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Video]):
            pytest.skip(
                f"{ApiClass.__name__} does not support videos. Skipping video tests.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text('Can you describe what you see in the video?')
        video_path = self._get_local_test_video_uri('flying.mp4')
        user_message.append_video(video_path)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert 'disc' in response or 'circular' in response or 'saucer' in response or 'rotor' in response or 'curved' in response or 'vehicle' in response or 'aircraft' in response or 'experimental' in response or 'hovering' in response or 'hover' in response or 'flying' in response

    def test_text_generation_with_video_url(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Video])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Video]):
            pytest.skip(
                f"{ApiClass.__name__} does not support videos. Skipping video tests.")

        print('\n' + default_model_name)

        video_url = 'https://ia903405.us.archive.org/27/items/archive-video-files/test.mp4'
        print(video_url)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you give a detailed description what you see in the following video?')
        user_message.append_video(video_url)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        if ApiClass.__name__ == 'OllamaApi':
            # If the ApiClass is OllamaAPi then the model is open source and might not be as strong as the others:
            assert 'image' in response
        else:
            assert 'rabbit' in response or 'bunny' in response or 'cartoon' in response or 'animated' in response
