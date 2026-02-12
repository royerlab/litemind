"""Tool use module for representing the results of tool executions.

A ``ToolUse`` captures the tool name, the arguments it was called with,
the execution result, and the identifier linking it back to the original
``ToolCall``.
"""

from typing import Any

from litemind.agent.messages.actions.action_base import ActionBase


class ToolUse(ActionBase):
    """Represents the result of executing a tool.

    Encapsulates the tool name, the arguments it was called with, the
    execution result, and the identifier linking it to the original
    ``ToolCall``.
    """

    def __init__(
        self, tool_name: str, arguments: dict = None, result: Any = None, id: str = None
    ):
        """
        Create a new tool use result.

        Parameters
        ----------
        tool_name : str
            The name of the tool that was executed.
        arguments : dict, optional
            The arguments the tool was called with. Defaults to an empty dict.
        result : Any, optional
            The result returned by the tool execution.
        id : str, optional
            The identifier matching the original tool call.
        """

        self.tool_name = tool_name
        self.arguments = arguments if arguments is not None else {}
        self.id = id

        # Add the result attribute
        self.result = result

    def __str__(self):
        """Return a concise string showing the tool call and its result.

        Returns
        -------
        str
            A string in the form ``tool_name(arg1=val1, ...)=result``.
        """
        args = self.arguments or {}
        arguments = ", ".join([f"{key}={value}" for key, value in args.items()])
        return f"{self.tool_name}({arguments})={self.result}"

    def __repr__(self):
        """Return a detailed string representation of the tool use.

        Returns
        -------
        str
            A string in the form
            ``ToolUse(tool=..., arguments=..., result=..., id=...)``.
        """
        # Format self.argument as a string of key=value pairs:
        args = self.arguments or {}
        arguments = ", ".join([f"{key}={value}" for key, value in args.items()])
        return f"ToolUse(tool={self.tool_name}, arguments={arguments}, result={str(self.result)}, id={self.id})"

    def pretty_string(self):
        """
        Return a pretty string representation of the tool use.

        Returns
        -------
        str
            A pretty string representation of the tool use.
        """
        args = self.arguments or {}
        arguments = ", ".join([f"{key}={value}" for key, value in args.items()])
        return f"{self.tool_name}({arguments}) -> {str(self.result)} "
