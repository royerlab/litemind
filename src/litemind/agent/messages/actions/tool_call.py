from litemind.agent.messages.actions.action_base import ActionBase


class ToolCall(ActionBase):

    def __init__(self, tool_name: str, arguments: dict = None, id: str = None):
        """
        Object that encapsulates information about a tool call.

        Parameters
        ----------
        tool_name: str
            The name of the tool used.
        arguments
            The arguments used with the tool.
        id:str
            The id of the tool use message

        """

        self.tool_name = tool_name
        self.arguments = arguments
        self.id = id

    def __str__(self):
        arguments = ", ".join(
            [f"{key}={value}" for key, value in self.arguments.items()]
        )
        return f"{self.tool_name}({arguments})"

    def __repr__(self):
        arguments = ", ".join(
            [f"{key}={value}" for key, value in self.arguments.items()]
        )
        return f"ToolUse(tool={self.tool_name}, arguments={arguments}, id={self.id})"

    def pretty_string(self):
        """
        Return a pretty string representation of the tool call.

        Returns
        -------
        str
            A pretty string representation of the tool call.
        """
        arguments = ", ".join(
            [f"{key}={value}" for key, value in self.arguments.items()]
        )
        return f"{self.tool_name}({arguments}) "
