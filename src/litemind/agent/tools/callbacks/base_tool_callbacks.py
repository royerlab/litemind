from abc import ABC
from typing import Any


class BaseToolCallbacks(ABC):
    """
    Base class for tool callbacks.
    This class defines the interface for tool callbacks that can be used to
    monitor the lifecycle of tool usage, including start, end, and error handling.
    It is intended to be subclassed to implement specific callback behaviors.
    Subclasses should implement the methods to handle tool start, end, and error events.
    This class does not implement any functionality itself, but provides a structure
    for tool callbacks to follow.
    """

    def on_tool_start(self, tool: "BaseTool", *args, **kwargs) -> None:
        """
        Called when tool use is started.

        Parameters
        ----------
        tool: BaseTool
            Tool instance
        *args: Any
            Positional arguments for the tool
        **kwargs: Any
            Additional arguments for the tool

        """
        pass

    def on_tool_end(self, tool: "BaseTool", result: Any) -> None:
        """
        Called when tool use is ended.

        Parameters
        ----------
        tool: BaseTool
            Tool instance
        result: Any
            Result of the tool use

        """
        pass

    def on_tool_error(self, tool: "BaseTool", exception: Exception) -> None:
        """
        Called when tool use encounters an error.

        Parameters
        ----------
        tool: BaseTool
            Tool instance
        exception: Exception
            Exception that occurred during tool use

        """
        pass
