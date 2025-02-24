from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base class for a tool."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.parameters = {}

    def __repr__(self):
        return f"{self.name}(description={self.description})"

    def __str__(self):
        return self.__repr__()

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
