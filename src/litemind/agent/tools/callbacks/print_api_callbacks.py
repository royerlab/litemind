from pprint import pformat
from typing import Any

from arbol import aprint

from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks


class PrintApiCallbacks(BaseToolCallbacks):
    """
    A callback class that prints tool execution details to the console.
    This class inherits from BaseToolCallbacks and overrides methods to print
    information about tool execution, including start, end, and error events.
    It is useful for debugging and monitoring tool usage in an agent's workflow.
    """

    def __init__(
        self,
        print_on_tool_start: bool = False,
        print_on_tool_end: bool = False,
        print_on_tool_error: bool = True,
        **kwargs,
    ):
        """
        Initialize the PrintApiCallbacks with options to control printing behavior.
        This constructor sets up the callback to print messages when tools start,
        end, or encounter errors. It raises an error if any unknown keyword arguments

        Parameters
        ----------
        print_on_tool_start: bool
            Whether to print a message when a tool starts execution.
        print_on_tool_end: bool
            Whether to print a message when a tool ends execution.
        print_on_tool_error: bool
            Whether to print a message when a tool encounters an error.
        kwargs: dict
            Additional keyword arguments that are not used by this class.
        """
        super().__init__()

        # Set the printing options based on the provided parameters:
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
