import pytest

from litemind.agent.messages.actions.tool_use import ToolUse
from litemind.media.types.media_action import Action


class TestAction:

    @pytest.fixture
    def sample_tool_use(self):
        """Create a sample ToolUse object for testing."""
        return ToolUse(
            tool_name="calculator", arguments={"a": 1, "b": 2}, result=3, id="test-id"
        )

    def test_init_with_pydantic_model(self, sample_tool_use):
        """Test initializing Action with a valid Pydantic model."""
        # Using 'object' attribute_key since validation checks for it
        action = Action(action=sample_tool_use)
        assert action.action == sample_tool_use

    def test_init_invalid_none(self):
        """Test initialization with None raises ValueError."""
        with pytest.raises(ValueError, match="Object cannot be None"):
            Action(action=None)

    def test_init_with_non_action(self):
        """Test initialization with non-pydantic object raises ValueError."""
        with pytest.raises(ValueError, match="Object must be an ActionBase"):
            Action(action="not a model")

    def test_str(self, sample_tool_use):
        """Test the string representation of Action."""
        action = Action(action=sample_tool_use)
        result = str(action)
        assert isinstance(result, str)
        # Assuming Action.__str__ might use the class name
        assert "calculator(a=1, b=2)=3" in result

    def test_kwargs_are_stored(self, sample_tool_use):
        """Test that additional kwargs are stored as attributes."""
        action = Action(action=sample_tool_use, custom_attr="test_value")
        assert "custom_attr" in action.attributes
