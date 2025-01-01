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
        assert isinstance(models,
                          list), f"{ApiClass.__name__}.model_list() should return a list!"
        assert len(
            models) > 0, f"{ApiClass.__name__}.model_list() should not be empty!"

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

    def test_completion_with_toolset(self, ApiClass):
        """
        Test that the completion method can interact with a toolset.
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

        assert "2024-11-15" in response.text, (
            f"The response of {ApiClass.__name__} should contain the delivery date."
        )

    def test_completion_with_image_url(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.default_model(require_vision=True)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        user_message.append_image_uri(
            'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg')
        user_message.append_text('Can you describe what you see in the image?')
        messages.append(user_message)

        # Run agent:
        response = api_instance.completion(messages=messages,
                                           model_name=default_model_name)

        # Normalise response:
        response = str(response)

        # Check response:
        assert 'sepia' in response or 'chalkboard' in response

        print('\n' + response)

    def test_completion_with_image_path(self, ApiClass):
        api_instance = ApiClass()
        default_model_name = api_instance.default_model(require_vision=True)

        messages = []

        # System message:
        system_message = Message(role='system')
        system_message.append_text(
            'You are an omniscient all-knowing being called Ohmm')
        messages.append(system_message)

        # User message:
        user_message = Message(role='user')
        image_path = self._get_image_path('future.jpeg')
        user_message.append_image_path(image_path)
        user_message.append_text('Can you describe what you see?')
        messages.append(user_message)

        # Run agent:
        response = api_instance.completion(messages=messages,
                                           model_name=default_model_name)

        # Normalise response:
        response = str(response)

        # Check response:
        assert 'robot' in response or 'futuristic' in response or 'sky' in response

        print('\n' + response)

    def test_has_vision_support(self, ApiClass):
        """
        Test has_vision_support's return type (True/False).
        We simply call it to ensure it's not blowing up.
        """
        api_instance = ApiClass()

        # Get a model that does not necessarily support vision:
        default_model_name = api_instance.default_model()
        support = api_instance.has_vision_support(model_name=default_model_name)
        # We don't assert True or False globally, because some implementations
        # might support it, others might not. Just ensure it's a boolean.
        assert isinstance(support, bool), (
            f"{ApiClass.__name__}.has_vision_support() should return a bool."
        )

        # Then we get a model that does not necessarily support vision:
        default_model_name = api_instance.default_model(require_vision=True)
        assert api_instance.has_vision_support(model_name=default_model_name)

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

    def test_describe_image_if_supported(self, ApiClass):
        """
        Test describe_image if the implementation supports vision.
        If not, we skip it.
        """
        api_instance = ApiClass()

        default_model_name = api_instance.default_model(require_vision=True)

        if not api_instance.has_vision_support(default_model_name):
            pytest.skip(
                f"{ApiClass.__name__} does not support vision. Skipping image test.")

        image_path = self._get_image_path('future.jpeg')

        description = api_instance.describe_image(image_path,
                                                  model_name=default_model_name)

        assert isinstance(description, str), (
            f"{ApiClass.__name__}.describe_image() should return a string!"
        )
        assert len(description) > 0, (
            f"{ApiClass.__name__}.describe_image() should return a non-empty string!"
        )
        # You can also add content checks if you want to verify the actual text.

    def _get_image_path(self, image_name: str):
        import os

        # Get the directory of the current file
        current_dir = os.path.dirname(__file__)

        # Combine the two to get the absolute path
        absolute_path = os.path.join(current_dir,
                                     os.path.join('images/', image_name))

        aprint(absolute_path)
        return absolute_path
