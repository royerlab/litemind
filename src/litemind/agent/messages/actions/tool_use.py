from typing import Any

from litemind.agent.messages.actions.action_base import ActionBase


class ToolUse(ActionBase):

    def __init__(
        self, tool_name: str, arguments: dict = None, result: Any = None, id: str = None
    ):
        """
        Object that encapsulates information about tool use.

        Parameters
        ----------
        tool_name: str
            The name of the tool used.
        arguments
            The arguments used with the tool.
        result: Any
            The result of the tool use.
        id:str
            The id of the tool use message

        """

        self.tool_name = tool_name
        self.arguments = arguments
        self.id = id

        # Add the result attribute
        self.result = result

    def __str__(self):
        arguments = ", ".join(
            [f"{key}={value}" for key, value in self.arguments.items()]
        )
        return f"{self.tool_name}({arguments})={self.result}"

    def __repr__(self):
        # Format self.argument as a stringof key=value pairs:
        arguments = ", ".join(
            [f"{key}={value}" for key, value in self.arguments.items()]
        )
        return f"ToolUse(tool={self.tool_name}, arguments={arguments}, result={str(self.result)}, id={self.id})"

    def pretty_string(self):
        """
        Return a pretty string representation of the tool use.

        Returns
        -------
        str
            A pretty string representation of the tool use.
        """
        arguments = ", ".join(
            [f"{key}={value}" for key, value in self.arguments.items()]
        )
        return f"{self.tool_name}({arguments}) -> {str(self.result)} "
