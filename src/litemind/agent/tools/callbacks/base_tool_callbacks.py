from abc import ABC
from typing import Any


class BaseToolCallbacks(ABC):
    """Base class for tool lifecycle callbacks.

    Subclass this to implement custom behavior when tools start, finish,
    encounter errors, or report intermediate activity. All methods are
    no-ops by default, so subclasses only need to override the events
    they care about.
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

    def on_tool_activity(self, tool: "BaseTool", activity_type: str, **kwargs) -> Any:
        """
        Called when a tool reports intermediate activity.

        Useful for monitoring multi-step tool executions or accessing
        internal progress information.

        Parameters
        ----------
        tool : BaseTool
            The active tool instance.
        activity_type : str
            A label describing the activity (e.g., "processing", "waiting").
        **kwargs
            Additional context about the activity.

        Returns
        -------
        Any
            An optional value that can be passed back to the tool.
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
