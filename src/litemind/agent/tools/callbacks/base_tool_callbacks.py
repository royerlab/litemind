from abc import ABC
from typing import Any


class BaseToolCallbacks(ABC):

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
