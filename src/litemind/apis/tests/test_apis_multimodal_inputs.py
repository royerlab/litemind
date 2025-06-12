import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.agent.messages.message import Message
from litemind.apis.base_api import ModelFeatures
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_video import Video
from litemind.ressources.media_resources import MediaResources


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
class TestBaseApiImplementationsMultimodalInputs(MediaResources):
    """
    A tests suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    These tests are for the multimodal inputs of the API.
    """

    def test_text_generation_with_image_url(self, api_class):
        api_instance = api_class()

        # Get the default model name:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image]
        )

        # Skip tests if the model does not support images
        if not default_model_name:
            pytest.skip(
                f"{api_class.__name__} does not support images. Skipping image tests."
            )

        print("\n" + default_model_name)

        image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
        print(image_url)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text("Can you describe what you see in the image?")
        user_message.append_image(image_url)
        messages.append(user_message)

        # Generate text:
        response = api_instance.generate_text(
            messages=messages, model_name=default_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Check response:
        assert (
            "sepia" in response
            or "chalkboard" in response
            or "Einstein" in response
            or "black-and-white" in response
            or "photograph" in response
        )

    def test_text_generation_with_png_image_path(self, api_class):
        api_instance = api_class()

        # Get the default model name:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image]
        )

        # Skip tests if the model does not support images
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support images. Skipping image tests."
            )

        # Print the model name:
        print("\n" + default_model_name)

        # Skip tests if the model does not support images
        if not default_model_name or not api_instance.has_model_support_for(
            model_name=default_model_name,
            features=[ModelFeatures.TextGeneration, ModelFeatures.Image],
        ):
            pytest.skip(
                f"{api_class.__name__} does not support images. Skipping image tests."
            )

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text("Can you describe what you see in the image?")
        image_path = self.get_local_test_image_uri("python.png")
        user_message.append_image(image_path)

        messages.append(user_message)

        # Generate text:
        response = api_instance.generate_text(
            messages=messages, model_name=default_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Check response:
        if api_class.__name__ == "OllamaApi":
            # open source models are typically not yet strong enough.
            assert "blue" in response or "logo" in response
        else:
            assert "snake" in response or "serpent" in response or "python" in response

    def test_text_generation_with_jpg_image_path(self, api_class):
        api_instance = api_class()

        # Get the best model for text generation and image:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image]
        )

        # Skip tests if the model does not support images
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support images. Skipping image tests."
            )

        print("\n" + default_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text("Can you describe what you see in the image?")
        image_path = self.get_local_test_image_uri("future.jpeg")
        user_message.append_image(image_path)

        messages.append(user_message)

        # Generate text:
        response = api_instance.generate_text(
            messages=messages, model_name=default_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Check response:
        assert "robot" in response or "futuristic" in response or "sky" in response

    def test_text_generation_with_webp_image_path(self, api_class):
        api_instance = api_class()

        # Get the best model for text generation and image:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image]
        )

        # Skip tests if the model does not support images
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support images. Skipping image tests."
            )

        print("\n" + default_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text("Can you describe what you see in the image?")
        image_path = self.get_local_test_image_uri("beach.webp")
        user_message.append_image(image_path)

        messages.append(user_message)

        # Generate text:
        response = api_instance.generate_text(
            messages=messages, model_name=default_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Check response:
        assert "beach" in response or "palm" in response or "sunset" in response

    def test_text_generation_with_gif_image_path(self, api_class):
        api_instance = api_class()

        # Get the best model for text generation and image:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image]
        )

        # Skip tests if the model does not support images
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support images. Skipping image tests."
            )

        print("\n" + default_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text("Can you describe what you see in the image?")
        image_path = self.get_local_test_image_uri("field.gif")
        user_message.append_image(image_path)

        messages.append(user_message)

        # Generate text:
        response = api_instance.generate_text(
            messages=messages, model_name=default_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Check response:
        assert "field" in response or "blue" in response or "stars" in response

    def test_text_generation_with_multiple_images(self, api_class):
        api_instance = api_class()

        # Get the best model for text generation and image:
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image]
        )

        # Skip tests if the model does not support images
        if default_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support images. Skipping image tests."
            )

        print("\n" + default_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text(
            "Can you compare these two images? What is similar and what is different?"
        )
        cat_image_path = self.get_local_test_image_uri("cat.jpg")
        panda_image_path = self.get_local_test_image_uri("panda.jpg")
        user_message.append_image(cat_image_path)
        user_message.append_image(panda_image_path)

        messages.append(user_message)

        # Generate text:
        response = api_instance.generate_text(
            messages=messages, model_name=default_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # If api_class is OllamaApi then the model is open source and might not be as strong as the others:
        if api_class.__name__ == "OllamaApi":
            assert "panda" in response

        # Check response:
        assert (
            ("animal" in response or "character" in response or "subject" in response)
            and ("cat" in response or "creature" in response)
            and "panda" in response
        )

    def test_text_generation_with_audio_path(self, api_class):
        api_instance = api_class()

        # Get the default model name:
        best_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration], media_types=[Audio]
        )

        # Skip tests if the model does not support audio
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support audio. Skipping audio tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text("Can you describe what you heard in the audio file?")
        audio_path = self.get_local_test_audio_uri("harvard.wav")
        user_message.append_audio(audio_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Get the response as a string:
        response = str(response).lower()

        # Check response:
        assert (
            "smell" in response
            or "ham" in response
            or "beer" in response
            or "reading" in response
            or "test passage" in response
        )

    def test_text_generation_with_audio_url(self, api_class):
        api_instance = api_class()

        # Get the default model name:
        best_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration], media_types=[Audio]
        )

        # Skip tests if the model does not support audio
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support audio. Skipping audio tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text(
            "Can you describe in detail what is said the following audio file?"
        )
        user_message.append_audio(
            "https://salford.figshare.com/ndownloader/files/14630270"
        )

        messages.append(user_message)

        # Generate text:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Check response:
        assert "canoe" in response or "chicken" in response or "hours" in response

    def test_text_generation_with_video_path(self, api_class):
        api_instance = api_class()

        # Get the best model for text generation and video:
        best_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration], media_types=[Video]
        )

        # Skip tests if the model does not support videos
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support videos. Skipping video tests."
            )

        print("\n" + best_model_name)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text("Can you describe what you see in the video?")
        video_path = self.get_local_test_video_uri("flying.mp4")
        user_message.append_video(video_path)
        messages.append(user_message)

        # Generate text:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Check response:
        assert (
            "disc" in response
            or "circular" in response
            or "saucer" in response
            or "rotor" in response
            or "curved" in response
            or "vehicle" in response
            or "aircraft" in response
            or "experimental" in response
            or "hovering" in response
            or "hover" in response
            or "flying" in response
            or "spacecraft" in response
        )

    def test_text_generation_with_video_url(self, api_class):
        api_instance = api_class()

        # Get the best model for text generation and video:
        best_model_name = api_instance.get_best_model(
            features=[ModelFeatures.TextGeneration], media_types=[Video]
        )

        # Skip tests if the model does not support videos
        if best_model_name is None:
            pytest.skip(
                f"{api_class.__name__} does not support videos. Skipping video tests."
            )

        print("\n" + best_model_name)

        video_url = (
            "https://ia803405.us.archive.org/27/items/archive-video-files/test.mp4"
        )
        print(video_url)

        messages = []

        # System message:
        system_message = Message(role="system")
        system_message.append_text(
            "You are an omniscient all-knowing being called Ohmm"
        )
        messages.append(system_message)

        # User message:
        user_message = Message(role="user")
        user_message.append_text(
            "Can you give a detailed description what you see in the following video?"
        )
        user_message.append_video(video_url)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text(
            messages=messages, model_name=best_model_name
        )

        for message in messages:
            print(message)

        # Get the last, and possibly only, message in response:
        response = response[-1]

        # Check response:
        if api_class.__name__ == "OllamaApi":
            # If the ApiClass is OllamaAPi then the model is open source and might not be as strong as the others:
            assert (
                "image" in response
                or "video" in response
                or "cartoon" in response
                or "animated" in response
                or "rabbit" in response
                or "bunny" in response
                or "character" in response
            )
        else:
            assert (
                "rabbit" in response
                or "bunny" in response
                or "cartoon" in response
                or "animated" in response
            )
