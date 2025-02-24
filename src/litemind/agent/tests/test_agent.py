import pytest
from arbol import aprint

from litemind.agent.agent import Agent
from litemind.agent.messages.conversation import Conversation
from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.apis.model_features import ModelFeatures
from litemind.apis.tests.base_test import API_IMPLEMENTATIONS


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_agent_text_with_openai(api_class):

    # Create OpenAI API object:
    api = api_class()

    # Check that at least one model is available for text generation, if not, skip test:
    if not api.has_model_support_for(features=ModelFeatures.TextGeneration):
        aprint(f"Skipping test for {api_class.__name__} as no text model is available.")
        return

    aprint(f"Default model: {api.get_best_model()}")

    # Create agent:
    agent = Agent(api=api)

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


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_message_image(api_class):

    # Create OpenAI API object:
    api = api_class()

    # Check that at least one model is available for text generation and image input, if not, skip test:
    if not api.has_model_support_for(
        features=[ModelFeatures.TextGeneration, ModelFeatures.Image]
    ):
        aprint(
            f"Skipping test for {api_class.__name__} as no text gen and image input model is available."
        )
        return

    # Get best image model:
    image_model_name = api.get_best_model(
        features=[
            ModelFeatures.TextGeneration,
            ModelFeatures.Image,
            ModelFeatures.Tools,
        ]
    )

    # Print image model:
    aprint(f"Image model: {image_model_name}")

    # Create agent:
    agent = Agent(api=api, model=image_model_name)

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

    # Run agent:
    reply = agent(conversation)

    # Check response:
    assert len(agent.conversation) == 3
    assert "sepia" in reply or "photograph" in reply or "chalkboard" in reply
