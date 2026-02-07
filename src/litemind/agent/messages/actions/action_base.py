from abc import ABC, abstractmethod


class ActionBase(ABC):
    """Abstract base class for tool-related actions (calls and results).

    Subclasses represent either a tool call request (from the assistant)
    or a tool use result (from tool execution).
    """

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
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
