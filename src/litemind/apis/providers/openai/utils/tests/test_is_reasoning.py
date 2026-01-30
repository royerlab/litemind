import pytest

from litemind.apis.providers.openai.utils.is_reasoning import (
    does_openai_model_support_temperature,
    is_openai_reasoning_model,
)


@pytest.mark.parametrize(
    "model_name, expected",
    [
        # All known reasoning models (o-series)
        ("o1", True),
        ("o1-mini", True),
        ("o1-preview", True),
        ("o1-low", True),
        ("o1-high", True),
        ("o3", True),
        ("o3-mini", True),
        ("o3-mini-high", True),
        ("o3-low", True),
        ("o3-medium", True),
        ("o3-high", True),
        ("o4-mini", True),
        ("o4-mini-high", True),
        ("o4-mini-low", True),
        ("o5", True),
        ("o5-mini", True),
        # Non-reasoning examples (should NOT match)
        ("gpt-4.5", False),
        ("gpt-4.5-preview", False),
        ("gpt-4o", False),
        ("gpt-4o-mini", False),
        ("davinci-002", False),
        ("gpt-5", False),
        ("gpt-5.2", False),
        # Edge cases - should NOT match models that happen to contain o1/o3/etc.
        ("modelo3", False),  # o3 is not at start
        ("gpt-4o1", False),  # o1 is not a separate token
        ("proto3-test", False),  # o3 is not at start
    ],
)
def test_is_reasoning_model(model_name, expected):
    """Ensure is_reasoning_model correctly recognises o-series models only."""
    assert is_openai_reasoning_model(model_name) is expected


@pytest.mark.parametrize(
    "model_name, expected",
    [
        # Models that SUPPORT temperature
        ("gpt-4o", True),
        ("gpt-4o-mini", True),
        ("gpt-4.5", True),
        ("gpt-4.5-preview", True),
        ("gpt-4.1", True),
        ("gpt-4.1-nano", True),
        ("gpt-4-turbo", True),
        ("davinci-002", True),
        # O-series models do NOT support temperature
        ("o1", False),
        ("o1-mini", False),
        ("o1-preview", False),
        ("o3", False),
        ("o3-mini", False),
        ("o3-high", False),
        ("o3-medium", False),
        ("o4-mini", False),
        ("o4-mini-high", False),
        ("o4-mini-low", False),
        ("o5", False),
        # GPT-5.x series do NOT support temperature
        ("gpt-5", False),
        ("gpt-5-pro", False),
        ("gpt-5-mini", False),
        ("gpt-5.1", False),
        ("gpt-5.1-codex", False),
        ("gpt-5.2", False),
        ("gpt-5.2-pro", False),
        ("gpt-5.2-mini", False),
        # Edge cases - models that look like gpt-5 but aren't
        ("gpt-50", True),  # Not gpt-5.x pattern
        ("gpt-5x", True),  # Not gpt-5.x pattern (no dot or hyphen after 5)
    ],
)
def test_does_openai_model_support_temperature(model_name, expected):
    """Ensure does_openai_model_support_temperature correctly identifies models."""
    assert does_openai_model_support_temperature(model_name) is expected
