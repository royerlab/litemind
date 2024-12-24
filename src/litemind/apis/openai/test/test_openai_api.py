import pytest
from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.apis.openai.openai_api_key import is_openai_api_key_available


@pytest.mark.skipif(not is_openai_api_key_available(),
                    reason="requires OpenAI key to run and availability of a vision model")
def test_openai_completion():

    openai_api = OpenAIApi()

    # check API key:
    assert openai_api.check_api_key()

    # Check model list is not None or empty:
    assert openai_api.model_list() is not None and len(openai_api.model_list()) > 0

    # Prepare messages:
    messages = []

    # System message:
    system_message = Message(role='system')
    system_message.append_text('You are an omniscient all-knowing being called Ohmm')
    messages.append(system_message)

    # User message:
    user_message = Message(role='user')
    user_message.append_text('Who are you?')
    messages.append(user_message)

    # Run agent:
    response = openai_api.completion(messages)

    # Check response:
    assert 'I am Ohmm' in str(response)

    print('\n'+str(response))



@pytest.mark.skipif(not is_openai_api_key_available(), reason="requires OpenAI API key to run")
def test_openai_api_completion_with_toolset():
    # Define a simple tool function that the assistant can call
    def get_delivery_date(order_id: str) -> str:
        """Fetch the delivery date for a given order ID."""
        return "2024-11-15"  # Return a sample delivery date

    # Initialize ToolSet and add the tool
    toolset = ToolSet()
    toolset.add_function_tool(get_delivery_date, "Fetch the delivery date for a given order ID")

    # Initialize the OpenAIApi with the API key from environment variables
    api = OpenAIApi()

    # Create a user message that prompts the assistant to retrieve the delivery date
    user_message = Message(role="user", text="When will my order order_12345 be delivered?")
    messages = [user_message]

    # Run the completion method with the toolset
    response = api.completion(messages=messages, toolset=toolset)

    # Check the response content to see if it contains the expected function result
    assert "2024-11-15" in response.text, "The response should contain the expected delivery date."




@pytest.mark.skipif(not is_openai_api_key_available(),
                    reason="requires OpenAI key to run and availability of a vision model")
def test_openai_image():

    # Create OpenAI API object:
    openai_api = OpenAIApi()

    # Check API key:
    messages = []

    # System message:
    system_message = Message(role='system')
    system_message.append_text('You are an omniscient all-knowing being called Ohmm')
    messages.append(system_message)

    # User message:
    user_message = Message(role='user')
    user_message.append_image_url(
        'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg')
    user_message.append_text('Can you describe what you see?')
    messages.append(user_message)

    # Run agent:
    response = openai_api.completion(messages)

    # Check response:
    assert 'sepia' in str(response)

    print('\n'+str(response))



@pytest.mark.skipif(not is_openai_api_key_available(),
                    reason="requires OpenAI key to run and availability of a vision model")
def test_vision():

    openai = OpenAIApi()

    image_1_path = _get_image_path('python.png')

    description_1 = openai.describe_image(image_1_path)

    print(description_1)

    assert ('logo' in description_1 or 'snake' in description_1 or 'serpent' in description_1) and 'Python' in description_1

    image_2_path = _get_image_path('future.jpeg')

    description_2 = openai.describe_image(image_2_path)

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

