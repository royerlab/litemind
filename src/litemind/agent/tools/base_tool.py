"""Abstract base class defining the interface for all Agent tools.

This module provides ``BaseTool``, the root class for function tools,
agent tools, and built-in tools. It defines a uniform interface with a
name, description, arguments schema, and callback manager, and routes
execution through ``__call__`` so that lifecycle callbacks are always
invoked.
"""

from abc import ABC, abstractmethod
from typing import Any

from litemind.agent.tools.callbacks.tool_callback_manager import ToolCallbackManager


class BaseTool(ABC):
    """Abstract base class for all tools usable by an Agent.

    Tools wrap callable functionality (functions, sub-agents, or built-in
    API features) behind a uniform interface. Each tool has a name,
    description, arguments schema, and a callback manager for lifecycle
    monitoring.
    """

    def __init__(self, name: str, description: str):
        """
        Initialize the base tool.

        Parameters
        ----------
        name : str
            The name of the tool. Whitespace is stripped and collapsed.
        description : str
            A description of what the tool does. A trailing period is
            added if not present.
        """

        # Store the name, make sure to strip any leading or trailing whitespace and any spaces in between:
        self.name = "".join(name.strip().split())

        # Make sure that the description finishes with a period:
        self.description = description.strip()
        if not self.description.endswith("."):
            self.description += "."

        # Initialize the arguments schema
        self.arguments_schema = {}

        # Initialise Callback manager:
        self.callbacks = ToolCallbackManager()

    def __repr__(self):
        """Return a detailed string representation of the tool.

        Returns
        -------
        str
            A string of the form ``name(description=...)``.
        """
        return f"{self.name}(description={self.description})"

    def __str__(self):
        """Return the string representation of the tool.

        Delegates to ``__repr__``.

        Returns
        -------
        str
            The same string returned by ``__repr__``.
        """
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
    def _execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool with the provided arguments.

        .. warning::
            Do not call this method directly. Use ``__call__`` instead
            to ensure callbacks are invoked.

        Parameters
        ----------
        *args
            Positional arguments for the tool.
        **kwargs
            Keyword arguments for the tool.

        Returns
        -------
        Any
            The result of the tool execution.
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool as a callable, invoking lifecycle callbacks.

        This is the primary entry point for running a tool. It fires
        ``on_tool_start`` before execution, ``on_tool_end`` after success,
        and ``on_tool_error`` if an exception is raised. Only keyword
        arguments are accepted.

        Parameters
        ----------
        *args
            Positional arguments are **not** supported and will raise
            ``ValueError``.
        **kwargs
            Keyword arguments forwarded to the underlying ``_execute``
            method.

        Returns
        -------
        Any
            The result produced by ``_execute``.

        Raises
        ------
        ValueError
            If positional arguments are provided.
        Exception
            Any exception raised by ``_execute`` is re-raised after the
            ``on_tool_error`` callback is invoked.
        """

        if args:
            raise ValueError(
                "Positional arguments are not supported. Use keyword arguments instead."
            )

        # Call the callbacks for tool's start:
        self.callbacks.on_tool_start(self, *args, **kwargs)

        try:
            # Run the tool's execute method and capture the result:
            result: Any = self._execute(*args, **kwargs)
        except Exception as e:
            # Call the callbacks for tool's error:
            self.callbacks.on_tool_error(self, e)
            raise e

        # Call the callbacks for tool's end:
        self.callbacks.on_tool_end(self, result)

        return result
