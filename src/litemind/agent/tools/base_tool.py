from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for a tool."""

    def __init__(self, name: str, description: str):

        # Store the name, make sure to strip any leading or trailing whitespace and any spaces in between:
        self.name = "".join(name.strip().split())

        # Make sure that the description finishes with a period:
        self.description = description.strip()
        if not self.description.endswith("."):
            self.description += "."

        # Initialize the arguments schema
        self.arguments_schema = {}

    def __repr__(self):
        return f"{self.name}(description={self.description})"

    def __str__(self):
        return self.__repr__()

    def is_builtin(self) -> bool:
        """
        Check if the tool is a built-in tool.

        Returns
        -------
        bool
            True if the tool is a built-in tool, False otherwise.
        """
        from litemind.agent.tools.builtin_tools.builtin_tool import BuiltinTool

        return isinstance(self, BuiltinTool)

    @abstractmethod
    def pretty_string(self):
        """
        Return a pretty string representation of the tool.

        Returns
        -------
        str
            A pretty string representation of the tool.
        """
        pass

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool with the provided arguments."""
        pass

    def __call__(self, *args, **kwargs):
        """
        Allow calling the tool as a function.

        Parameters
        ----------
        *args
            Positional arguments (not supported)
        **kwargs
            Arbitrary keyword arguments to pass to the tool

        Returns
        -------
        Any
            The result of the tool function.

        """

        if args:
            raise ValueError(
                "Positional arguments are not supported. Use keyword arguments instead."
            )
        return self.execute(*args, **kwargs)
