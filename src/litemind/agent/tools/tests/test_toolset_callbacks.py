import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.agent.tools.agent_tool import AgentTool
from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks
from litemind.agent.tools.function_tool import FunctionTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.model_features import ModelFeatures


class DummyCallback(BaseToolCallbacks):
    def __init__(self):
        self.started = False
        self.ended = False
        self.errored = False

    def on_tool_start(self, tool, *args, **kwargs):
        self.started = True

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
