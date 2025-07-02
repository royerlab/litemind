import pytest

from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.callbacks.print_api_callbacks import PrintApiCallbacks


class DummyTool(BaseTool):
    def __init__(self, name="dummy", description="A dummy tool."):
        super().__init__(name, description)

    def pretty_string(self):
        return "DummyTool"

    def _execute(self, *args, **kwargs):
        return "result"


@pytest.fixture
def print_callbacks():
    return PrintApiCallbacks(
        print_on_tool_start=True, print_on_tool_end=True, print_on_tool_error=True
    )


@pytest.fixture
def dummy_tool():
    return DummyTool()


def test_on_tool_start(print_callbacks, dummy_tool, capsys):
    print_callbacks.on_tool_start(dummy_tool, foo="bar")
    captured = capsys.readouterr()
    assert (
        f"Tool Start: {dummy_tool.name} with args: () and kwargs: {{'foo': 'bar'}}"
        in captured.out
    )


def test_on_tool_end(print_callbacks, dummy_tool, capsys):
    print_callbacks.on_tool_end(dummy_tool, result="some result")
    captured = capsys.readouterr()
    assert f"Tool End: {dummy_tool.name} with result: some result" in captured.out


def test_on_tool_error(print_callbacks, dummy_tool, capsys):
    exc = Exception("error message")
    print_callbacks.on_tool_error(dummy_tool, exc)
    captured = capsys.readouterr()
    assert (
        f"Tool Error: {dummy_tool.name} with exception: error message" in captured.out
    )
