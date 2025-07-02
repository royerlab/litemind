from pprint import pformat
from typing import Any

from arbol import aprint

from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks


class PrintApiCallbacks(BaseToolCallbacks):

    def __init__(
        self,
        print_on_tool_start: bool = False,
        print_on_tool_end: bool = False,
        print_on_tool_error: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.print_on_tool_start = print_on_tool_start
        self.print_on_tool_end = print_on_tool_end
        self.print_on_tool_error = print_on_tool_error

        if kwargs:
            raise ValueError(f"Unknown arguments: {pformat(kwargs)}")

    def on_tool_start(self, tool: BaseTool, *args, **kwargs) -> None:
        aprint(f"Tool Start: {tool.name} with args: {args} and kwargs: {kwargs}")

    def on_tool_end(self, tool: BaseTool, result: Any) -> None:
        aprint(f"Tool End: {tool.name} with result: {result}")

    def on_tool_error(self, tool: BaseTool, exception: Exception) -> None:
        aprint(f"Tool Error: {tool.name} with exception: {exception}")
