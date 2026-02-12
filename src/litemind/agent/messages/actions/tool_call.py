"""Tool call module for representing LLM-initiated tool execution requests.

A ``ToolCall`` captures the tool name, arguments, and a unique identifier
that the framework uses to match the call to its corresponding ``ToolUse``
result.
"""

from litemind.agent.messages.actions.action_base import ActionBase


class ToolCall(ActionBase):
    """Represents a request from the LLM to execute a tool.

    Encapsulates the tool name, arguments, and a unique identifier that
    links this call to its corresponding ``ToolUse`` result.
    """

    def __init__(self, tool_name: str, arguments: dict = None, id: str = None):
        """
        Create a new tool call.

        Parameters
        ----------
        tool_name : str
            The name of the tool to call.
        arguments : dict, optional
            The arguments for the tool call. Defaults to an empty dict.
        id : str, optional
            A unique identifier for this tool call.
        """

        self.tool_name = tool_name
        self.arguments = arguments if arguments is not None else {}
        self.id = id

    def __str__(self):
        """Return a concise string showing the tool call as a function invocation.

        Returns
        -------
        str
            A string in the form ``tool_name(arg1=val1, arg2=val2)``.
        """
        args = self.arguments or {}
        arguments = ", ".join([f"{key}={value}" for key, value in args.items()])
        return f"{self.tool_name}({arguments})"

    def __repr__(self):
        """Return a detailed string representation of the tool call.

        Returns
        -------
        str
            A string in the form
            ``ToolCall(tool=..., arguments=..., id=...)``.
        """
        args = self.arguments or {}
        arguments = ", ".join([f"{key}={value}" for key, value in args.items()])
        return f"ToolCall(tool={self.tool_name}, arguments={arguments}, id={self.id})"

    def pretty_string(self):
        """
        Return a pretty string representation of the tool call.

        Returns
        -------
        str
            A pretty string representation of the tool call.
        """
        args = self.arguments or {}
        arguments = ", ".join([f"{key}={value}" for key, value in args.items()])
        return f"{self.tool_name}({arguments}) "
