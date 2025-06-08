import pytest

from litemind.agent.messages.message import Message
from litemind.apis.model_features import ModelFeatures
from litemind.apis.providers.ollama.ollama_api import OllamaApi
from litemind.ressources.media_resources import MediaResources


def test_compare_images_with_separate_messages():
    api_instance = OllamaApi()

    # Skip test if Ollama is not available:
    if not api_instance.check_availability_and_credentials():
        pytest.skip(f"{OllamaApi.__name__} is not available. Skipping image tests.")

    # Get the best model for text generation and images:
    default_model_name = api_instance.get_best_model(
        [ModelFeatures.TextGeneration, ModelFeatures.Image]
    )

    # Skip tests if the model does not support images
    if default_model_name is None:
        pytest.skip(
            f"{OllamaApi.__name__} does not support images. Skipping image tests."
        )

    print("\n" + default_model_name)

    messages = []

    # System message:
    system_message = Message(role="system")
    system_message.append_text("You are an omniscient all-knowing being called Ohmm")
    messages.append(system_message)

    # User message for the first image:
    user_message_1 = Message(role="user")
    user_message_1.append_text("Here is the first image.")
    cat_image_path = MediaResources.get_local_test_image_uri("cat.jpg")
    user_message_1.append_image(cat_image_path)
    messages.append(user_message_1)

    # User message for the second image:
    user_message_2 = Message(role="user")
    user_message_2.append_text("Here is the second image.")
    panda_image_path = MediaResources.get_local_test_image_uri("panda.jpg")
    user_message_2.append_image(panda_image_path)
    messages.append(user_message_2)

    # User message asking for comparison:
    user_message_3 = Message(role="user")
    user_message_3.append_text(
        "Can you compare these two images? What is similar and what is different?"
    )
    messages.append(user_message_3)

    # Generate text:
    response = api_instance.generate_text(
        messages=messages, model_name=default_model_name
    )

    # Print messages:
    for message in messages:
        print(message)

    # Get the last response message:
    response = response[-1].lower()

    # Check response:
    assert (
        ("animal" in response or "character" in response or "subject" in response)
        and ("cat" in response or "creature" in response)
        and "panda" in response
    )
