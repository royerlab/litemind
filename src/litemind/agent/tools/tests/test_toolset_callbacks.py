from typing import Any

import pytest

from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.tools.toolset import ToolSet


# Corrected DummyCallback class
class DummyCallback(BaseToolCallbacks):
    def __init__(self):
        self.started = False
        self.activity = ""
        self.ended = False
        self.errored = False

    def on_tool_start(self, tool, *args, **kwargs):
        self.started = True

    def on_tool_activity(self, tool, message, **kwargs):
        self.activity += (
            f"Tool Activity: {tool.name} with message: {message} and kwargs: {kwargs}"
        )

    def on_tool_end(self, tool, result):
        self.ended = True

    def on_tool_error(self, tool, exception):
        self.errored = True


def test_add_and_remove_tool_callback():
    toolset = ToolSet()
    func_tool = FunctionTool(lambda x: x, "desc")
    toolset.add_tool(func_tool)
    callback = DummyCallback()

    # Add callback and check it's registered
    toolset.add_tool_callback(callback)
    assert callback in func_tool.callbacks

    # Remove callback and check it's unregistered
    toolset.remove_tool_callback(callback)
    assert callback not in func_tool.callbacks


def test_tool_callback_invocation():
    toolset = ToolSet()
    func_tool = FunctionTool(lambda x: x + 1, "desc")
    toolset.add_tool(func_tool)
    callback = DummyCallback()
    toolset.add_tool_callback(callback)

    # Trigger tool execution to invoke callbacks
    func_tool(x=1)
    assert callback.started
    assert callback.ended


def test_tool_callback_error_invocation():
    toolset = ToolSet()

    def error_func(x):
        raise ValueError("fail")

    func_tool = FunctionTool(error_func, "desc")
    toolset.add_tool(func_tool)
    callback = DummyCallback()
    toolset.add_tool_callback(callback)

    with pytest.raises(ValueError):
        func_tool(x=1)
    assert callback.started
    assert callback.errored


def test_tool_callback_on_activity():
    toolset = ToolSet()

    def some_function(x):
        return x * 2

    class SomeFunctionTool(FunctionTool):
        def __init__(self):
            super().__init__(func=some_function, description="Some function")

        def _execute(self, *args, **kwargs) -> Any:
            # Simulate some intermediate activity
            self.callbacks.on_tool_activity(self, "stuff", info="Intermediate activity")
            return super()._execute(*args, **kwargs)

    # Create an instance of the function tool:
    func_tool = SomeFunctionTool()

    # Add the tool to the toolset:
    toolset.add_tool(func_tool)

    # Create a dummy callback and add it to the toolset:
    callback = DummyCallback()
    toolset.add_tool_callback(callback)

    # Call the tool to trigger the activity callback:
    func_tool(x=5)

    # Check that the activity callback was invoked:
    assert "Intermediate activity" in callback.activity
    assert "Tool Activity" in callback.activity
    assert "some_function" in callback.activity
