"""Base module for tool-related actions.

Defines the abstract ``ActionBase`` class from which ``ToolCall`` and
``ToolUse`` are derived. An action captures either a request from the
LLM to execute a tool or the result returned after executing it.
"""

from abc import ABC, abstractmethod


class ActionBase(ABC):
    """Abstract base class for tool-related actions (calls and results).

    Subclasses represent either a tool call request (from the assistant)
    or a tool use result (from tool execution).
    """

    @abstractmethod
    def __str__(self):
        """Return a concise string representation of the action.

        Returns
        -------
        str
            A human-readable summary of the action.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """Return a detailed string representation of the action.

        Returns
        -------
        str
            A string that includes the action type and all relevant fields.
        """
        pass

    @abstractmethod
    def pretty_string(self):
        """
        Return a human-readable string representation of the action.

        Returns
        -------
        str
            A formatted string describing the action.
        """
        pass
