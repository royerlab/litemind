import pytest
from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
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


@pytest.mark.parametrize("ApiClass", API_IMPLEMENTATIONS)
class TestBaseApiImplementations:
    """
    A test suite that runs the same tests on each ApiClass
    implementing the abstract BaseApi interface.
    """

    def test_check_api_key(self, ApiClass):
        """
        Test that check_api_key() returns True (or behaves appropriately)
        for each implementation.
        """
        api_instance = ApiClass()
        assert api_instance.check_api_key() is True, (
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

    def test_has_image_support(self, ApiClass):
        """
        Test has_image_support's return type (True/False).
        We simply call it to ensure it's not blowing up.
        """
        api_instance = ApiClass()

        # Get a model that does not necessarily support images:
        default_model_name = api_instance.default_model()
        support = api_instance.has_image_support(model_name=default_model_name)
        # We don't assert True or False globally, because some implementations
        # might support it, others might not. Just ensure it's a boolean.
        assert isinstance(support, bool), (
            f"{ApiClass.__name__}.has_image_support() should return a bool."
        )

        # Then we get a model that does necessarily support vision:
        default_model_name = api_instance.default_model(require_images=True)

        # if a model is returned, check if it supports images:
        if default_model_name:
            assert api_instance.has_image_support(model_name=default_model_name)

        # Let's list models that support vision:
        models = api_instance.model_list()
        for model in models:
            if api_instance.has_image_support(model):
                aprint(model)

    def test_has_audio_support(self, ApiClass):
        """
        Test has_audio_support's return type (True/False).
        We simply call it to ensure it's not blowing up.
        """
        api_instance = ApiClass()

        # Get a model that does not necessarily support audio:
        default_model_name = api_instance.default_model()
        support = api_instance.has_image_support(model_name=default_model_name)
        # We don't assert True or False globally, because some implementations
        # might support it, others might not. Just ensure it's a boolean.
        assert isinstance(support, bool), (
            f"{ApiClass.__name__}.has_audio_support() should return a bool."
        )

        # Then we get a model that does necessarily support audio:
        default_model_name = api_instance.default_model(require_audio=True)

        # If a model is returned, check if it supports audio:
        if default_model_name:
            assert api_instance.has_audio_support(model_name=default_model_name)

        # Let's list models that support audio:
        models = api_instance.model_list()
        for model in models:
            if api_instance.has_audio_support(model):
                aprint(model)

    def test_has_tool_support(self, ApiClass):
        """
        Tests whether the model supports tool usage. Some models may or may not.
        Just verify that the method returns a bool and possibly confirm
        certain models do or do not support tools.
        """
        api_instance = ApiClass()

        # Check default model:
        default_model_name = api_instance.default_model()
        support = api_instance.has_tool_support(model_name=default_model_name)

        assert isinstance(support, bool), (
            f"{ApiClass.__name__}.has_tool_support() should return a bool!"
        )

        tool_model_name = api_instance.default_model(require_tools=True)

        if default_model_name:
            assert api_instance.has_tool_support(model_name=tool_model_name)

        # Let's list models that support tools:
        models = api_instance.model_list()
        for model in models:
            if api_instance.has_tool_support(model):
                aprint(model)

    def test_max_num_input_token(self, ApiClass):
        """
        Test that max_num_input_token returns a valid integer.
        """

        api_instance = ApiClass()
        default_model_name = api_instance.default_model()
        max_tokens = api_instance.max_num_input_tokens(
            model_name=default_model_name)
        assert isinstance(max_tokens, int), (
            f"{ApiClass.__name__}.max_num_input_token() should return an int!"
        )
        assert max_tokens > 0, (
            f"{ApiClass.__name__}.max_num_input_token() should be a positive integer!"
        )

    def test_max_num_output_tokens(self, ApiClass):
        """
        Test that max_num_output_tokens() returns a valid integer.
        """
        api_instance = ApiClass()
        default_model_name = api_instance.default_model()
        max_tokens = api_instance.max_num_output_tokens(
            model_name=default_model_name)
        assert isinstance(max_tokens, int), (
            f"{ApiClass.__name__}.max_num_output_tokens() should return an int!"
        )
        assert max_tokens > 0, (
            f"{ApiClass.__name__}.max_num_output_tokens() should be a positive integer!"
        )

    def test_completion_simple(self, ApiClass):
        """
        Test a simple completion with the default or a known good model.
        """
        api_instance = ApiClass()

        default_model_name = api_instance.default_model()

        messages = [
            Message(role="system",
                    text="You are an omniscient all-knowing being called Ohmm"),
            Message(role="user", text="Who are you?")
        ]

        # Some implementations let you omit `model_name` if they have a default.
        # If your API requires a model_name, supply it here:
        response = api_instance.completion(
            model_name=default_model_name,
            messages=messages,
            temperature=0.7
        )

        assert response.role == "assistant", (
            f"{ApiClass.__name__} completion should return an 'assistant' role."
        )
        assert "I am " in response.text, (
            f"Expected 'I am' in the output of {ApiClass.__name__}.completion()"
        )

    def test_completion_with_simple_toolset(self, ApiClass):
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

        default_model_name = api_instance.default_model(require_tools=True)

        user_message = Message(role="user",
                               text="When will my order order_12345 be delivered?")
        messages = [user_message]

        response = api_instance.completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        for message in messages:
            aprint(message)

        assert "2024-11-15" in response.text, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

    def test_completion_with_complex_toolset(self, ApiClass):
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

        default_model_name = api_instance.default_model(require_tools=True)

        messages = []

        user_message = Message(role="user",
                               text="When will my order 'order_12345' be delivered?")
        messages += [user_message]

        response = api_instance.completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        assert "2024-11-15" in response.text, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

        user_message = Message(role="user",
                               text="What is the name of product 1393?")
        messages += [user_message]

        response = api_instance.completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        assert "Olea Table" in response.text, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

        user_message = Message(role="user",
                               text="How many Olea Tables can I find in store 17?")
        messages += [user_message]

        response = api_instance.completion(
            model_name=default_model_name,  # or a specific model name if needed
            messages=messages,
            toolset=toolset
        )

        for message in messages:
            aprint(message)

        assert "42" in response.text, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

    def test_completion_with_image_url(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.default_model(require_images=True)
        aprint(default_model_name)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_text('Can you describe what you see in the image?')
        user_message.append_image_url(
            'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg')
        messages.append(user_message)

        # Run agent:
        response = api_instance.completion(messages=messages,
                                           model_name=default_model_name)

        # Normalise response:
        response = str(response)

        # Check response:
        assert 'sepia' in response or 'chalkboard' in response

        print('\n' + response)

    def test_completion_with_png_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.default_model(require_images=True)
        aprint(default_model_name)

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
        user_message.append_image_path(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.completion(messages=messages,
                                           model_name=default_model_name)

        # Normalise response:
        response = str(response)

        # Check response:
        assert 'snake' in response or 'python' in response

        print('\n' + response)

    def test_completion_with_jpg_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.default_model(require_images=True)
        aprint(default_model_name)

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
        user_message.append_image_path(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.completion(messages=messages,
                                           model_name=default_model_name)

        # Normalise response:
        response = str(response)

        # Check response:
        assert 'robot' in response or 'futuristic' in response or 'sky' in response

        print('\n' + response)

    def test_completion_with_webp_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.default_model(require_images=True)
        aprint(default_model_name)

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
        user_message.append_image_path(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.completion(messages=messages,
                                           model_name=default_model_name)

        # Normalise response:
        response = str(response)

        # Check response:
        assert 'beach' in response or 'palm' in response or 'sunset' in response

        print('\n' + response)

    def test_completion_with_gif_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.default_model(require_images=True)
        aprint(default_model_name)

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
        user_message.append_image_path(image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.completion(messages=messages,
                                           model_name=default_model_name)

        # Normalise response:
        response = str(response)

        # Check response:
        assert 'field' in response or 'blue' in response or 'stars' in response

        print('\n' + response)

    def test_completion_with_multiple_images(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.default_model(require_images=True)

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
        user_message.append_image_path(cat_image_path)
        user_message.append_image_path(panda_image_path)

        messages.append(user_message)

        # Run agent:
        response = api_instance.completion(messages=messages,
                                           model_name=default_model_name)

        # Normalise response:
        response = str(response)

        # Check response:
        assert (
                           'animals' in response or 'characters' in response) and 'cat' in response and 'panda' in response

        print('\n' + response)

    def test_describe_image_if_supported(self, ApiClass):
        """
        Test describe_image if the implementation supports images.
        If not, we skip it.
        """
        api_instance = ApiClass()

        try:

            default_model_name = api_instance.default_model(require_images=True)

            if not api_instance.has_image_support(default_model_name):
                pytest.skip(
                    f"{ApiClass.__name__} does not support images. Skipping image test.")

            image_path = self._get_local_test_image_uri('future.jpeg')

            description = api_instance.describe_image(image_path,
                                                      model_name=default_model_name)

            aprint(description)

            assert isinstance(description, str), (
                f"{ApiClass.__name__}.describe_image() should return a string!"
            )
            assert len(description) > 0, (
                f"{ApiClass.__name__}.describe_image() should return a non-empty string!"
            )
            # You can also add content checks if you want to verify the actual text.
        except:
            # If exception happened then test failed:
            assert False

    def _get_local_test_image_uri(self, image_name: str):
        import os

        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)

        # Combine the two to get the absolute path
        absolute_path = os.path.join(current_dir,
                                     os.path.join('images/', image_name))

        uri = 'file://' + absolute_path

        aprint(uri)
        return uri
