"""Unit test for Agent handling of None API responses."""

from unittest.mock import MagicMock, patch

from litemind.agent.agent import Agent
from litemind.agent.messages.message import Message


def test_agent_call_with_none_response():
    """Test that Agent.__call__ handles None response without crashing.

    Regression test: when the API returns None (e.g. due to an error),
    conversation.extend(None) used to raise TypeError.
    """
    # Create a mock API:
    mock_api = MagicMock()
    mock_api.get_best_model.return_value = "mock-model"
    mock_api.has_model_support_for.return_value = True
    mock_api.list_models.return_value = ["mock-model"]
    mock_api.generate_text.return_value = None

    # Create agent with mock API:
    agent = Agent(api=mock_api)
    agent.append_system_message("You are a test agent.")

    # This should not crash even though generate_text returns None:
    response = agent("Hello")

    # Response should be None:
    assert response is None

    # Conversation should have system + user messages only (no None appended):
    assert len(agent.conversation) == 2
    assert agent.conversation[0].role == "system"
    assert agent.conversation[1].role == "user"
