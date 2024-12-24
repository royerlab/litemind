from unittest.mock import MagicMock

import pytest
from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.ollama.ollama_api import OllamaApi


@pytest.fixture
def ollama_api():
    """Create an instance of OllamaApi with a mocked client."""
    return OllamaApi()

@pytest.fixture
def model_name(ollama_api):
    # Search for first laama model in list:
    for model in ollama_api.model_list():
        if 'llama' in model:
            return model


@pytest.fixture
def vision_model_name(ollama_api):
    # Search llava model in list:
    for model in ollama_api.model_list():
        if 'llava' in model:
            return model


# Tests
def test_check_api_key_success(ollama_api):
    """Test check_api_key with a successful response."""
    assert ollama_api.check_api_key() is True


def test_has_vision_support(ollama_api):
    """Test has_vision_support always returns False."""
    assert ollama_api.has_vision_support() is False


def test_max_num_input_token(ollama_api, model_name):
    """Test max_num_input_token returns the correct limit."""
    assert ollama_api.max_num_input_token(model_name) >= 2500


def test_completion_success(ollama_api, model_name):
    """Test completion with a valid response."""
    mock_response = MagicMock()
    mock_response.message.content = "Hello, user!"

    messages = [Message(role="user", text="Hi there!")]
    response = ollama_api.completion(messages=messages,
                                     model_name=model_name)

    assert response.role == "assistant"
    assert "Hello" in response.text


def test_completion_toolset_not_supported(ollama_api):
    """Test completion raises an error if toolset is provided."""
    messages = [Message(role="user", text="Hi there!")]
    toolset = ToolSet()

    with pytest.raises(NotImplementedError,
                       match="Tool usage is not supported by Ollama."):
        ollama_api.completion(messages=messages, model_name="llama3.2",
                              toolset=toolset)


def test_list_models(ollama_api):
    """Test list_models returns a list of models."""
    model_list = ollama_api.model_list()

    assert model_list is not None and len(model_list)>0

def test_vision(ollama_api, vision_model_name):

    image_1_path = _get_image_path('python.png')

    description_1 = ollama_api.describe_image(image_1_path, model_name=vision_model_name)

    print(description_1)

    assert ('logo' in description_1 or 'snake' in description_1 or 'serpent' in description_1) and 'Python' in description_1

    image_2_path = _get_image_path('future.jpeg')

    description_2 = ollama_api.describe_image(image_2_path, model_name=vision_model_name)

    print(description_2)

    assert ('futuristic' in description_2 or 'science fiction' in description_2 or 'robots' in description_2) and ('sunset' in description_2 or 'sunrise' in description_2 or 'landscape' in description_2)


def _get_image_path(image_name: str):
    import os

    # Get the directory of the current file
    current_dir = os.path.dirname(__file__)

    # Combine the two to get the absolute path
    absolute_path = os.path.join(current_dir, os.path.join('images/',image_name))

    aprint(absolute_path)
    return absolute_path