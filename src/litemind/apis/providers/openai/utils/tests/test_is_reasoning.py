import pytest

from litemind.apis.providers.openai.utils.is_reasoning import is_openai_reasoning_model


@pytest.mark.parametrize(
    "model_name, expected",
    [
        # All known reasoning models
        ("o1", True),
        ("o1-mini", True),
        ("o1-preview", True),
        ("o3", True),
        ("o3-mini", True),
        ("o3-mini-high", True),
        ("o4-mini", True),
        ("o4-mini-high", True),
        # Non-reasoning examples
        ("gpt-4.5", False),
        ("gpt-4.5-preview", False),
        ("gpt-4o", False),
        ("davinci-002", False),
    ],
)
def test_is_reasoning_model(model_name, expected):
    """Ensure is_reasoning_model correctly recognises o-series models only."""
    assert is_openai_reasoning_model(model_name) is expected
