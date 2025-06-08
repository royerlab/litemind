import pytest  # adjust the import

from litemind.apis.providers.google.utils.is_reasoning import is_gemini_reasoning_model


@pytest.mark.parametrize(
    "model_id, expected",
    [
        # Positive cases
        ("gemini-1.5-pro", True),
        ("gemini-1.5-flash", True),
        ("gemini-2.0-pro", True),
        ("gemini-2.0-flash-thinking-exp-01-21", True),
        ("gemini-2.5-pro-preview-05-06", True),
        ("gemini-2.5-flash", True),
        # Negative cases
        ("gemini-nano", False),
        ("gemini-1.0-pro", False),
        ("gpt-4o", False),
        ("o3", False),
    ],
)
def test_is_gemini_reasoning_model(model_id, expected):
    """Ensure only reasoning-ready Gemini models return True."""
    assert is_gemini_reasoning_model(model_id) is expected
