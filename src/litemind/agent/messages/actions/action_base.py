from abc import abstractmethod


class ActionBase:

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pretty_string(self):
        """
        Return a pretty string representation of the tool call.

        Returns
        -------
        str
            A pretty string representation of the tool call.
        """
        pass
