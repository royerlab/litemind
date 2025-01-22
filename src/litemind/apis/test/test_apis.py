from io import BytesIO
from pathlib import Path
from typing import Dict

import pytest
from PIL import Image
from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.anthropic.anthropic_api import AnthropicApi
from litemind.apis.base_api import ModelFeatures
from litemind.apis.google.google_api import GeminiApi
from litemind.apis.ollama.ollama_api import OllamaApi
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.apis.test.utils.levenshtein import levenshtein_distance
from litemind.utils.normalise_uri_to_local_file_path import \
    uri_to_local_file_path

# Put all your implementations in this list:
API_IMPLEMENTATIONS = [
    OpenAIApi,
    OllamaApi,
    AnthropicApi,
    GeminiApi
]


@pytest.mark.parametrize("ApiClass", API_IMPLEMENTATIONS)
class TestBaseApiImplementations:
    """
    A test suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    """

    def test_availability_and_credentials(self, ApiClass):
        """
        Test that check_api_key() returns True (or behaves appropriately)
        for each implementation.
        """
        api_instance = ApiClass()
        assert api_instance.check_availability_and_credentials() is True, (
            f"{ApiClass.__name__}.check_api_key() should be True!"
        )

    def test_model_list(self, ApiClass):
        """
        Test that model_list() returns a non-empty list.
        """
        api_instance = ApiClass()
        models = api_instance.model_list()
        for model in models:
            aprint(model)
        assert isinstance(models,
                          list), f"{ApiClass.__name__}.model_list() should return a list!"
        assert len(
            models) > 0, f"{ApiClass.__name__}.model_list() should not be empty!"

    def test_has_has_model_support_for(self, ApiClass):

        api_instance = ApiClass()

        for feature in ModelFeatures:

            aprint(f"Checking support for feature: {feature} in {ApiClass}")

            # Get a model that does not necessarily support feature:
            default_model_name = api_instance.get_best_model()

            print('\n' + default_model_name)

            # Check if the model supports the feature:
            support = api_instance.has_model_support_for(
                model_name=default_model_name, features=feature)

            # We don't assert True or False globally, because some implementations
            # might support it, others might not. Just ensure it's a boolean.
            assert isinstance(support, bool), (
                f"{ApiClass.__name__}.has_model_support_for({feature}) should return a bool."
            )

            # Then we get a model that does necessarily support feature:
            default_model_name = api_instance.get_best_model(features=feature)

            # if a model is returned, check if it supports feature:
            if default_model_name:
                assert api_instance.has_model_support_for(
                    model_name=default_model_name, features=feature)
            else:
                aprint(f"No model in {ApiClass} supports feature: {feature}")

            # Let's list models that support feature:
            models = api_instance.model_list()
            for model in models:
                if api_instance.has_model_support_for(model_name=model,
                                                      features=feature):
                    aprint(model)

            print(
                f"Checked support for feature: {feature} in {ApiClass}: all good!")

    def test_get_model_features(self, ApiClass):
        """
        Test that get_model_features() returns a non-empty list of features for all API models.
        """

        api_instance = ApiClass()
        models = api_instance.model_list()
        for model in models:
            features = api_instance.get_model_features(model_name=model)

            print(f"\nModel: {model} Features: {features}")

            assert isinstance(features, list), (
                f"{ApiClass.__name__}.get_model_features() should return a list!"
            )
            assert len(features) > 0, (
                f"{ApiClass.__name__}.get_model_features() should not be empty!"
            )
            for feature in features:
                assert isinstance(feature, ModelFeatures), (
                    f"{ApiClass.__name__}.get_model_features() should return a list of ModelFeatures!"
                )

    def test_max_num_input_token(self, ApiClass):
        """
        Test that max_num_input_token returns a valid integer.
        """

        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model()

        print('\n' + default_model_name)

        max_input_tokens = api_instance.max_num_input_tokens(
            model_name=default_model_name)

        print(f"\nMax input tokens: {max_input_tokens}")

        assert isinstance(max_input_tokens, int), (
            f"{ApiClass.__name__}.max_num_input_token() should return an int!"
        )
        assert max_input_tokens > 4096, (
            f"{ApiClass.__name__}.max_num_input_token() should be a positive integer above 4096!"
        )

    def test_max_num_output_tokens(self, ApiClass):
        """
        Test that max_num_output_tokens() returns a valid integer.
        """
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model()

        print('\n' + default_model_name)

        max_output_tokens = api_instance.max_num_output_tokens(
            model_name=default_model_name)

        print(f"\nMax output tokens: {max_output_tokens}")

        assert isinstance(max_output_tokens, int), (
            f"{ApiClass.__name__}.max_num_output_tokens() should return an int!"
        )
        assert max_output_tokens > 0, (
            f"{ApiClass.__name__}.max_num_output_tokens() should be a positive integer!"
        )

    def test_text_generation_simple(self, ApiClass):
        """
        Test a simple completion with the default or a known good model.
        """
        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)

        print('\n' + default_model_name)

        messages = [
            Message(role="system",
                    text="You are an omniscient all-knowing being called Ohmm"),
            Message(role="user", text="Who are you?")
        ]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,
            messages=messages,
            temperature=0.7
        )

        print('\n' + str(response))

        assert response.role == "assistant", (
            f"{ApiClass.__name__} completion should return an 'assistant' role."
        )
        assert "I am " in response, (
            f"Expected 'I am' in the output of {ApiClass.__name__}.completion()"
        )

    def test_text_generation_prefill(self, ApiClass):
        """
        Test a simple completion with prefilled answer
        """
        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)

        print('\n' + default_model_name)

        messages = [
            Message(role="system",
                    text="You are very good at Math."),
            Message(role="user",
                    text="What is 12+17 equal to? Please give the answer in parentheses."),
            Message(role="assistant", text="(")
        ]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,
            messages=messages,
            temperature=0.0
        )

        print('\n' + str(response))

        assert response.role == "assistant", (
            f"{ApiClass.__name__} completion should return an 'assistant' role."
        )
        assert "29" in response, (
            f"Expected '29' in the output of {ApiClass.__name__}.completion()"
        )
        # Better the string should just be "29)" and not "29) " or "29 )":
        if not str(response[0]).startswith("29)"):
            aprint(
                "Model does not support strict prefill behaviour: The response should start with '29)'")

    def test_text_generation_with_simple_toolset(self, ApiClass):
        """
        Test that the completion method can interact with a simple toolset.
        """

        # If you have a function-based tool usage pattern:
        def get_delivery_date(order_id: str) -> str:
            """Fetch the delivery date for a given order ID."""
            return "2024-11-15"

        toolset = ToolSet()
        toolset.add_function_tool(
            get_delivery_date,
            "Fetch the delivery date for a given order ID"
        )

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration])

        print('\n' + default_model_name)

        user_message = Message(role="user",
                               text="When will my order order_12345 be delivered?")
        messages = [user_message]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        print('\n')
        for message in messages:
            aprint(message)

        assert "2024-11-15" in response, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

    def test_text_generation_with_complex_toolset(self, ApiClass):
        """
        Test that the completion method can interact with a complex toolset.
        """

        def get_delivery_date(order_id: str) -> str:
            """Fetch the delivery date for a given order ID."""
            return "2024-11-15"

        def get_product_name_from_id(product_id: int) -> str:
            """Fetch the delivery date for a given order ID."""

            product_table = {1393: 'Olea Table', 84773: 'Fluff Phone'}

            return product_table[product_id]

        def get_product_supply_per_store(store_id: int, product_id: int) -> str:
            """Fetch the delivery date for a given order ID."""

            # Return a random integer:
            return 42

        toolset = ToolSet()
        toolset.add_function_tool(
            get_delivery_date,
            "Fetch the delivery date for a given order ID"
        )
        toolset.add_function_tool(
            get_product_name_from_id,
            "Fetch the product name for a given product ID"
        )
        toolset.add_function_tool(
            get_product_supply_per_store,
            "Fetch the number of items available of a given product id at a given store."
        )

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.Tools, ModelFeatures.TextGeneration])

        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Tools]):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation and tools. Skipping test.")

        messages = []

        user_message = Message(role="user",
                               text="When will my order 'order_12345' be delivered?")
        messages += [user_message]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        # Check that we have 2 messages in the list:
        assert len(messages) == 2, (
            'We should have two messages in the list. The user message and the response message.')

        assert "2024-11-15" in response, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

        user_message = Message(role="user",
                               text="What is the name of product 1393?")
        messages += [user_message]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        assert "Olea Table" in response, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

        user_message = Message(role="user",
                               text="How many Olea Tables can I find in store 17?")
        messages += [user_message]

        response = api_instance.generate_text_completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        print('\n')
        for message in messages:
            print(message)

        assert "42" in response, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

    def test_text_generation_with_image_url(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image test.")

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
        user_message.append_image(
            'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg')
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Check response:
        assert 'sepia' in response or 'chalkboard' in response or 'Einstein' in response

    def test_text_generation_with_png_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support images. Skipping image test.")

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
                f"{ApiClass.__name__} does not support images. Skipping image test.")

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
                f"{ApiClass.__name__} does not support images. Skipping image test.")

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
                f"{ApiClass.__name__} does not support images. Skipping image test.")

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
                f"{ApiClass.__name__} does not support images. Skipping image test.")

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
                f"{ApiClass.__name__} does not support audio. Skipping audio test.")

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
                f"{ApiClass.__name__} does not support audio. Skipping audio test.")

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
                f"{ApiClass.__name__} does not support videos. Skipping video test.")

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
                f"{ApiClass.__name__} does not support videos. Skipping video test.")

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
        user_message.append_text('Can you describe what you see in the video?')
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

    def test_text_generation_with_pdf_document(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Documents,
             ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Documents,
                          ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support documents. Skipping documents test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a highly qualified scientist with extensive experience in Microscopy, Biology and Bioimage processing and analysis.')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you write a review for the provided paper? Please break down your comments into major and minor comments.')
        doc_path = self._get_local_test_document_uri('intracktive_preprint.pdf')
        user_message.append_document(doc_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'microscopy' in response.lower() or 'biology' in response.lower()

        # Check if the response is detailed and follows instructions:

        if not ('inTRACKtive' in response and 'review' in response.lower()):
            # printout message that warns that the response miight lack in detail:
            aprint(
                "The response might lack in detail!! Please check the response for more details.")

    def test_text_generation_with_webpage(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            [ModelFeatures.TextGeneration, ModelFeatures.Documents,
             ModelFeatures.Image])
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=[ModelFeatures.TextGeneration, ModelFeatures.Documents,
                          ModelFeatures.Image]):
            pytest.skip(
                f"{ApiClass.__name__} does not support documents. Skipping documents test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a highly qualified scientist with extensive experience in Zebrafish biology.')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Can you summarise the contents of the webpage and what is known about this gene in zebrafish? Which tissue do you think this gene is expressed in?')
        user_message.append_document("https://zfin.org/ZDB-GENE-060606-1")

        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'ZFIN' in response or 'COMP' in response or 'zebrafish' in response

    def test_text_generation_with_json(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a computer program that can read complex json strings and understand what they contain.')
        messages.append(system_message)

        # complex and long test Json input:
        json_str = """
        {
            "name": "John Doe",
            "age": 30,
            "cars": {
                "car1": "Ford",
                "car2": "BMW",
                "car3": "Fiat"
            }
        }
        """

        # User message:
        user_message = Message(role='user')
        user_message.append_json(json_str)
        user_message.append_text('What is the age of John?')
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert '30' in response

    def test_text_generation_with_object(self, ApiClass):

        # Check if pydantic is installed otherwise skip test:
        try:
            from pydantic import BaseModel
        except ImportError:
            pytest.skip(
                "Pydantic is not installed. Skipping test."
            )

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a computer program that can read complex json strings and understand what they contain.')
        messages.append(system_message)

        # complex pydantic object  (derives from BaseModel) for testing:
        from pydantic import BaseModel
        class TestObject(BaseModel):
            name: str
            age: int
            cars: Dict[str, str]

        # Create instance of object and fill with details:
        test_object = TestObject(name='John Doe', age=30,
                                 cars={'car1': 'Ford', 'car2': 'BMW',
                                       'car3': 'Fiat'})

        # User message:
        user_message = Message(role='user')
        user_message.append_object(test_object)
        user_message.append_text('What is the age of John?')
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert '30' in response

    def test_text_generation_with_csv(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text('You are a highly qualified data scientist.')
        messages.append(system_message)

        # _get_local_test_table_uri
        table_path = self._get_local_test_table_uri('spreadsheet.csv')

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'List all items sold by rep Carl Jackson in the provided table.')
        user_message.append_table(table_path)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'Binders' in response or 'SAFCO' in response

    def test_text_generation_with_archive(self, ApiClass):

        api_instance = ApiClass()

        default_model_name = api_instance.get_best_model(
            ModelFeatures.TextGeneration)
        if not default_model_name or not api_instance.has_model_support_for(
                model_name=default_model_name,
                features=ModelFeatures.TextGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support text generation. Skipping test.")

        print('\n' + default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are a highly qualified historian and literature expert')
        messages.append(system_message)

        # _get_local_test_archive_uri
        archive_path = self._get_local_test_archive_uri('alexander.zip')

        # User message:
        user_message = Message(role='user')
        user_message.append_text(
            'Make a one paragraph summary of the provided material, plus a list of all documents provided.')
        user_message.append_archive(archive_path)
        messages.append(user_message)

        # Run agent:
        response = api_instance.generate_text_completion(messages=messages,
                                                         model_name=default_model_name)

        print('\n' + str(response))

        # Make sure that the answer is not empty:
        assert len(response) > 0, (
            f"{ApiClass.__name__}.completion() should return a non-empty string!"
        )

        # Check response details:
        assert 'Alexander' in response or 'Aristotle' in response

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
                f"{ApiClass.__name__} does not support images. Skipping image test.")

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

            # If exception happened then test failed:
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
                f"{ApiClass.__name__} does not support audio. Skipping audio test.")

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

            # If exception happened then test failed:
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
                f"{ApiClass.__name__} does not support videos. Skipping video test.")

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

            # If ApiClass.__name__ is 'OllamaApi', we relax the test to allow for more flexibility:
            if ApiClass.__name__ == 'OllamaApi':
                # Open source models are not yet strong enough to understand that a sequence of images is a video:
                assert 'elephant' in description.lower()
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

            # If exception happened then test failed:
            assert False

    def test_generate_audio(self, ApiClass):
        # Test audio generation with generate_audio:
        api_instance = ApiClass()
        audio_gen_model_name = api_instance.get_best_model(
            ModelFeatures.AudioGeneration)

        if not api_instance.has_model_support_for(
                model_name=audio_gen_model_name,
                features=ModelFeatures.AudioGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support audio generation. Skipping test.")

        aprint("Audio generation model name: ", audio_gen_model_name)

        # Define the text to generate audio from:
        text = "Success is the ability to go from one failure to another with no loss of enthusiasm"

        # Print the text:
        aprint(f"Text to generate audio from: '{text}'")

        # Generate the audio:
        generated_audio_uri = api_instance.generate_audio(
            model_name=audio_gen_model_name,
            text=text,
        )

        # save URI to file:
        generated_audio_file = uri_to_local_file_path(generated_audio_uri)

        # Convert the path string to a file to check its properties:
        generated_audio_file = Path(generated_audio_file)

        # Check that the file exists:
        assert generated_audio_file.exists(), (
            "The generated audio file should exist."
        )
        # Check the audio file length:
        assert generated_audio_file.stat().st_size > 0, (
            "The generated audio file should not be empty."
        )
        # Check that it is an MP3 file:
        assert generated_audio_file.suffix == '.mp3', (
            "The generated audio file should be an MP3 file."
        )

        # Transcribe the audio to check its contents:
        if api_instance.has_model_support_for(model_name=audio_gen_model_name,
                                              features=ModelFeatures.AudioTranscription):
            # Transcribe audio:
            transcription = api_instance.transcribe_audio(generated_audio_uri)

            # compute the edit distance between the two strings:

            edit_distance = levenshtein_distance(text, transcription)

            # Make sure that the distance is not more than the number of words in the text:
            assert edit_distance <= len(text.split()), (
                "The edit distance between the generated audio and the original text should be small."
            )

            print(f"Transcription: \n'{transcription}'")

    def test_generate_image(self, ApiClass):
        api_instance = ApiClass()
        image_gen_model_name = api_instance.get_best_model(
            ModelFeatures.ImageGeneration)

        if not api_instance.has_model_support_for(
                model_name=image_gen_model_name,
                features=ModelFeatures.ImageGeneration):
            pytest.skip(
                f"{ApiClass.__name__} does not support image generation. Skipping test.")

        aprint("Image generation model name: ", image_gen_model_name)

        prompt = "a white siamese cat"
        image_width = 1024
        image_height = 1024

        # Generate the image
        generated_image = api_instance.generate_image(
            model_name=image_gen_model_name,
            positive_prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            preserve_aspect_ratio=True,
            allow_resizing=True
        )

        assert isinstance(generated_image,
                          Image.Image), "The generated image should be a PIL Image instance."

        # Save the generated image to a BytesIO object
        image_bytes = BytesIO()
        generated_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        # Save image to temp PNG file using tempfile:
        import tempfile
        temp_image_path = tempfile.NamedTemporaryFile(suffix='.png').name
        generated_image.save(temp_image_path, format='PNG')

        # Convert filepath to URI:
        image_uri = 'file://' + temp_image_path

        # Get model with image support:
        image_model_name = api_instance.get_best_model(ModelFeatures.Image)
        aprint("Image model name: ", image_model_name)

        # If describe_image is available, use it to check the content of the generated image
        if image_model_name and api_instance.has_model_support_for(
                model_name=image_model_name, features=ModelFeatures.Image):
            # Ask model to describe image:
            description = api_instance.describe_image(
                model_name=image_model_name,
                image_uri=image_uri)

            print(f"Description of the generated image: \n{description}")

            # Asserts:
            assert isinstance(description,
                              str), "The description should be a string."
            assert len(description) > 0, "The description should not be empty."
            assert 'cat' in description.lower(), "The description should mention a 'cat'."
            assert (
                    'white' in description.lower() or 'blue' in description.lower()), "The description should mention 'white' or 'blue'."

    def test_text_embedding(self, ApiClass):
        """
        Test that text_embedding() returns a valid list of floats.
        """
        api_instance = ApiClass()

        embedding_model_name = api_instance.get_best_model(
            ModelFeatures.TextEmbeddings)

        # Skip test if the model does not support embeddings:
        if not embedding_model_name or not api_instance.has_model_support_for(
                model_name=embedding_model_name,
                features=ModelFeatures.TextEmbeddings):
            pytest.skip(
                f"{ApiClass.__name__} does not support embeddings. Skipping test.")

        texts = ["Hello, world!", "Testing embeddings."]
        embeddings = api_instance.embed_texts(texts=texts,
                                              model_name=embedding_model_name,
                                              dimensions=512)

        # Check that the returned embeddings are a list of length 2:
        assert isinstance(embeddings, list), "The embeddings should be a list."
        assert len(
            embeddings) == 2, "The embeddings should be a list of length 2."

        # Check that each embeddings are lists of floats:
        for embedding in embeddings:
            assert isinstance(embedding,
                              list), "Each embedding should be a list."
            assert len(
                embedding) == 512, "Each embedding should be of length 512 as reequested."

            for value in embedding:
                assert isinstance(value,
                                  float), "Each value in the embedding should be a float."

    def test_audio_embedding(self, ApiClass):
        """
        Test that text_embedding() returns a valid list of floats.
        """
        api_instance = ApiClass()

        embedding_model_name = api_instance.get_best_model(
            ModelFeatures.TextEmbeddings)

        # Skip test if the model does not support embeddings:
        if not embedding_model_name or not api_instance.has_model_support_for(
                model_name=embedding_model_name,
                features=ModelFeatures.TextEmbeddings):
            pytest.skip(
                f"{ApiClass.__name__} does not support embeddings. Skipping test.")

        texts = ["Hello, world!", "Testing embeddings."]
        embeddings = api_instance.embed_texts(texts=texts,
                                              model_name=embedding_model_name,
                                              dimensions=512)

        # Check that the returned embeddings are a list of length 2:
        assert isinstance(embeddings, list), "The embeddings should be a list."
        assert len(
            embeddings) == 2, "The embeddings should be a list of length 2."

        # Check that each embeddings are lists of floats:
        for embedding in embeddings:
            assert isinstance(embedding,
                              list), "Each embedding should be a list."
            assert len(
                embedding) == 512, "Each embedding should be of length 512 as reequested."

            for value in embedding:
                assert isinstance(value,
                                  float), "Each value in the embedding should be a float."

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
