"""Callback implementation that prints tool lifecycle events to the console.

This module provides ``PrintApiCallbacks``, a concrete
``BaseToolCallbacks`` subclass that logs tool start, activity, end,
and error events using the Arbol logging library. Each event type can
be independently enabled or disabled.
"""

from pprint import pformat
from typing import Any

from arbol import aprint

from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks


class PrintApiCallbacks(BaseToolCallbacks):
    """Tool callback that prints execution details to the console via Arbol.

    Useful for debugging and monitoring tool usage. Each lifecycle event
    (start, activity, end, error) can be independently enabled or disabled.
    """

    def __init__(
        self,
        print_on_tool_start: bool = False,
        print_on_tool_activity: bool = False,
        print_on_tool_end: bool = False,
        print_on_tool_error: bool = True,
        **kwargs,
    ):
        """
        Initialize the callback with per-event printing controls.

        Parameters
        ----------
        print_on_tool_start : bool
            Whether to print when a tool starts execution.
        print_on_tool_activity : bool
            Whether to print when a tool reports intermediate activity.
        print_on_tool_end : bool
            Whether to print when a tool finishes execution.
        print_on_tool_error : bool
            Whether to print when a tool encounters an error.
        **kwargs
            Rejected; raises ValueError if any unknown arguments are passed.

        Raises
        ------
        ValueError
            If unknown keyword arguments are provided.
        """
        super().__init__()

        # Set the printing options based on the provided parameters:
        self.print_on_tool_start = print_on_tool_start
        self.print_on_tool_activity = print_on_tool_activity
        self.print_on_tool_end = print_on_tool_end
        self.print_on_tool_error = print_on_tool_error

        if kwargs:
            raise ValueError(f"Unknown arguments: {pformat(kwargs)}")

    def on_tool_start(self, tool: BaseTool, *args, **kwargs) -> None:
        """Print tool start information if ``print_on_tool_start`` is enabled.

        Parameters
        ----------
        tool : BaseTool
            The tool that is starting execution.
        *args
            Positional arguments passed to the tool.
        **kwargs
            Keyword arguments passed to the tool.
        """
        aprint(f"Tool Start: {tool.name} with args: {args} and kwargs: {kwargs}")

    def on_tool_activity(self, tool: "BaseTool", activity_type: str, **kwargs) -> Any:
        """Print tool activity information if ``print_on_tool_activity`` is enabled.

        Parameters
        ----------
        tool : BaseTool
            The tool reporting activity.
        activity_type : str
            A label describing the activity.
        **kwargs
            Additional context about the activity.
        """
        aprint(f"Tool Activity: {tool.name} is {activity_type} with info: {kwargs}")

    def on_tool_end(self, tool: BaseTool, result: Any) -> None:
        """Print tool completion information if ``print_on_tool_end`` is enabled.

        Parameters
        ----------
        tool : BaseTool
            The tool that finished execution.
        result : Any
            The result produced by the tool.
        """
        aprint(f"Tool End: {tool.name} with result: {result}")

    def on_tool_error(self, tool: BaseTool, exception: Exception) -> None:
        """Print tool error information if ``print_on_tool_error`` is enabled.

        Parameters
        ----------
        tool : BaseTool
            The tool that encountered an error.
        exception : Exception
            The exception that was raised.
        """
        aprint(f"Tool Error: {tool.name} with exception: {exception}")
