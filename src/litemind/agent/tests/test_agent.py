import pytest
from arbol import aprint
from pydantic import BaseModel

from litemind import API_IMPLEMENTATIONS
from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_agent_with_just_text(api_class):
    # Create OpenAI API object:
    api = api_class()

    # Check that at least one model is available for text generation, if not, skip test:
    if not api.has_model_support_for(features=ModelFeatures.TextGeneration):
        aprint(f"Skipping test for {api_class.__name__} as no text model is available.")
        return

    aprint(f"Default model: {api.get_best_model()}")

    # Create agent:
    agent = Agent(api=api)

    # Add system message:
    agent.append_system_message("You are an omniscient all-knowing being called Ohmm")

    # Run agent, yu can pass a string directly to the agent:
    response = agent("Who are you?")

    # Print conversation
    aprint(agent.conversation)

    # Check that there is three messages in conversation:
    assert len(agent.conversation) == 3

    assert agent.conversation[0].role == "system"
    assert (
        "You are an omniscient all-knowing being called Ohmm" in agent.conversation[0]
    )
    assert agent.conversation[1].role == "user"
    assert "Who are you?" in agent.conversation[1]

    # Check that there is one message in the response:
    assert len(response) == 1

    # Extract last message from response:
    response = response[-1]

    # Check that response contains the system message:
    assert "Ohmm" in response


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_agent_with_just_text(api_class):
    # Create OpenAI API object:
    api = api_class()

    # Check that at least one model is available for text generation, if not, skip test:
    if not api.has_model_support_for(features=ModelFeatures.TextGeneration):
        aprint(f"Skipping test for {api_class.__name__} as no text model is available.")
        return

    aprint(f"Default model: {api.get_best_model()}")

    # 4. Using the Agent with a JSON Response Format
    class Weather(BaseModel):
        temperature: float
        condition: str
        humidity: float

    # Create agent:
    agent = Agent(api=api)

    # Add system message:
    agent.append_system_message(
        "You are a weather bot. Weather conditions must be: sunny, rainy, cloudy, snowy, or partly cloudy."
    )

    # Run agent, you can pass a string directly to the agent:
    response = agent(
        "What is the weather like in Paris? (if you don't know, imagine what your favorite weather would be like)",
        response_format=Weather,
    )
    print("Agent with JSON Response:", response)

    # Check that there is one message in the response:
    assert len(response) == 1

    # Extract last message from response:
    weather = response[-1][-1].get_content()

    # Check that response contains the system message:
    assert weather.humidity > 30
    assert weather.temperature > 10
    assert weather.condition.lower() in [
        "sunny",
        "rainy",
        "cloudy",
        "snowy",
        "partly cloudy",
    ]


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_agent_message_with_image(api_class):
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

    # Check that image model is available:
    if not image_model_name:
        aprint(
            f"Skipping test for {api_class.__name__} as no textgen + image + tools model is available."
        )
        return

    # Print image model:
    aprint(f"Image model: {image_model_name}")

    # Create agent:
    agent = Agent(api=api, model_name=image_model_name)

    agent.append_system_message("You are an omniscient all-knowing being called Ohmm")

    user_message = Message(role="user", text="Can you describe what you see?")
    user_message.append_image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Einstein_1921_by_F_Schmutzer_-_restoration.jpg/456px-Einstein_1921_by_F_Schmutzer_-_restoration.jpg"
    )

    # Run agent on message, you can pass a message object, or a list, directly to the agent:
    response = agent(user_message)

    # Check that agent conversation has three messages:
    assert len(agent.conversation) == 3

    # Check that there is one more message in the response:
    assert len(response) == 1

    # Extract the last message from the response:
    response = response[-1]

    # Check that the response contains the image description:
    assert "sepia" in response or "photograph" in response or "chalkboard" in response
