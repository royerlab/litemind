import pytest

from litemind.agent.agent import Agent
from litemind.agent.conversation import Conversation
from litemind.agent.message import Message
from litemind.apis.openai.openai_api import OpenAIApi
from litemind.apis.openai.openai_api_key import is_openai_api_key_available


@pytest.mark.skipif(not is_openai_api_key_available(),
                    reason="requires OpenAI key to run")
def test_agent_text_with_openai():
    # Create messages:
    system_message = Message(role='system',
                             text='You are an omniscient all-knowing being called Ohmm')
    user_message = Message(role='user', text='Who are you?')

    # Create conversation:
    conversation = Conversation()

    # Append messages:
    conversation += system_message
    conversation += user_message

    # Create OpenAI API object:
    api = OpenAIApi()

    # Create agent:
    agent = Agent(api=api)

    # Run agent:
    reply = agent(conversation)

    # Check response:
    assert len(agent.conversation) == 3
    assert 'I am Ohmm' in reply
    assert agent.conversation[0].role == 'system'
    assert agent.conversation[
               0].text == 'You are an omniscient all-knowing being called Ohmm'
    assert agent.conversation[1].role == 'user'
    assert agent.conversation[1].text == 'Who are you?'


@pytest.mark.skipif(not is_openai_api_key_available(),
                    reason="requires OpenAI key to run")
def test_message_image():
    # Create messages:
    system_message = Message(role='system',
                             text='You are an omniscient all-knowing being called Ohmm')
    user_message = Message(role='user', text='Can you describe what you see?')
    user_message.append_image_url(
        'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg')

    # Create conversation:
    conversation = Conversation()

    # Append messages:
    conversation += system_message
    conversation += user_message

    # Check conversation length and contents:
    assert len(conversation) == 2
    assert conversation[0].role == 'system'
    assert conversation[
               0].text == 'You are an omniscient all-knowing being called Ohmm'
    assert conversation[1].role == 'user'
    assert conversation[1].text == 'Can you describe what you see?'
    assert len(conversation[1].image_uris) == 1
    assert 'upload' in conversation[1].image_uris[0]

    # Create OpenAI API object:
    api = OpenAIApi()

    # Create agent:
    agent = Agent(api=api)

    # Run agent:
    reply = agent(conversation)

    # Check response:
    assert len(agent.conversation) == 3
    assert 'sepia' in reply or 'photograph' in reply
