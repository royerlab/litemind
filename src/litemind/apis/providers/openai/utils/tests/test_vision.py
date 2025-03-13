import pytest

from litemind.apis.providers.openai.utils.vision import has_vision_support


# Vision-capable models
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4o",
        "gpt-4o-2024-11-20",
        "gpt-4o-2024-08-06",
        "gpt-4o-2024-05-13",
        "chatgpt-4o-latest",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
    ],
)
def test_vision_capable_models(model_name):
    """Test models that should support vision capabilities"""
    assert has_vision_support(model_name) is True


# Non-vision models
@pytest.mark.parametrize(
    "model_name",
    [
        "gpt-4",
        "gpt-4-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-instruct",
        "non-existent-model",
        "invalid-model-name",
    ],
)
def test_non_vision_models(model_name):
    """Test models that should not support vision capabilities"""
    assert has_vision_support(model_name) is False


def test_invalid_model_name():
    """Test handling of invalid model names"""
    with pytest.raises(ValueError):
        has_vision_support("")
    with pytest.raises(ValueError):
        has_vision_support("   ")
