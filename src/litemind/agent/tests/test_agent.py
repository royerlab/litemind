import pytest
from arbol import aprint

from litemind.agent.agent import Agent
from litemind.agent.messages.conversation import Conversation
from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.apis.model_features import ModelFeatures
from litemind.apis.providers.openai.openai_api import OpenAIApi
from litemind.apis.providers.openai.utils.openai_api_key import (
    is_openai_api_key_available,
)


@pytest.mark.skipif(
    not is_openai_api_key_available(), reason="requires OpenAI key to run"
)
def test_agent_text_with_openai():
    # Create messages:
    system_message = Message(
        role="system", text="You are an omniscient all-knowing being called Ohmm"
    )
    user_message = Message(role="user", text="Who are you?")

    # Create conversation:
    conversation = Conversation()

    # Append messages:
    conversation += system_message
    conversation += user_message

    # Create OpenAI API object:
    api = OpenAIApi()

    aprint(f"Default model: {api.get_best_model()}")

    # Create agent:
    agent = Agent(api=api)

    # Run agent:
    reply = agent(conversation)

    aprint(conversation)

    # Check response:
    assert len(agent.conversation) == 3
    assert "I am Ohmm" in reply
    assert agent.conversation[0].role == "system"
    assert (
        "You are an omniscient all-knowing being called Ohmm" in agent.conversation[0]
    )
    assert agent.conversation[1].role == "user"
    assert "Who are you?" in agent.conversation[1]


@pytest.mark.skipif(
    not is_openai_api_key_available(), reason="requires OpenAI key to run"
)
def test_message_image():
    # Create messages:
    system_message = Message(
        role="system", text="You are an omniscient all-knowing being called Ohmm"
    )
    user_message = Message(role="user", text="Can you describe what you see?")
    user_message.append_image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
    )

    # Create conversation:
    conversation = Conversation()

    # Append messages:
    conversation += system_message
    conversation += user_message

    # Check conversation length and contents:
    assert len(conversation) == 2
    assert conversation[0].role == "system"
    assert "You are an omniscient all-knowing being called Ohmm" in conversation[0]
    assert conversation[1].role == "user"
    assert "Can you describe what you see?" in conversation[1]
    assert len(conversation[1]) == 2
    assert conversation[1][1].block_type == BlockType.Image

    # Create OpenAI API object:
    api = OpenAIApi()

    image_model_name = api.get_best_model(features=ModelFeatures.Image)

    aprint(f"Image model: {image_model_name}")

    # Create agent:
    agent = Agent(api=api, model=image_model_name)

    # Run agent:
    reply = agent(conversation)

    # Check response:
    assert len(agent.conversation) == 3
    assert "sepia" in reply or "photograph" in reply or "chalkboard" in reply
