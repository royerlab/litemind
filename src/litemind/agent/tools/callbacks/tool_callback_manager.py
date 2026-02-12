"""Manager that aggregates tool callbacks and dispatches lifecycle events.

This module provides ``ToolCallbackManager``, which holds a list of
``BaseToolCallbacks`` instances and fans out ``on_tool_start``,
``on_tool_end``, ``on_tool_activity``, and ``on_tool_error`` calls to
every registered callback.
"""

from typing import Any, List, Sequence, Union

from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks


class ToolCallbackManager(BaseToolCallbacks):
    """Manages a collection of tool callbacks and dispatches lifecycle events.

    Aggregates multiple ``BaseToolCallbacks`` instances and fans out
    ``on_tool_start``, ``on_tool_end``, ``on_tool_activity``, and
    ``on_tool_error`` calls to all registered callbacks.
    """

    def __init__(self):
        """Initialize the manager with an empty callback list."""
        self.callbacks: List[BaseToolCallbacks] = []

    def add_callback(self, callback: BaseToolCallbacks) -> None:
        """
        Register a callback if not already present.

        Parameters
        ----------
        callback : BaseToolCallbacks
            The callback to add.
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def add_callbacks(
        self, callbacks_: Union[Sequence[BaseToolCallbacks], "ApiCallbackManager"]
    ) -> None:
        """
        Register multiple callbacks at once.

        Parameters
        ----------
        callbacks_ : Union[Sequence[BaseToolCallbacks], ApiCallbackManager]
            A sequence of callbacks or another callback manager whose
            callbacks should be added.
        """
        if isinstance(callbacks_, BaseToolCallbacks):
            self.callbacks.extend(callbacks_.callbacks)
        elif isinstance(callbacks_, Sequence):
            for callback in callbacks_:
                if isinstance(callback, BaseToolCallbacks):
                    self.callbacks.append(callback)

    def remove_callback(self, callback: BaseToolCallbacks) -> None:
        """
        Remove a callback if it is registered.

        Parameters
        ----------
        callback : BaseToolCallbacks
            The callback to remove.
        """
        if callback in self.callbacks:
            # Remove the callback if it exists
            self.callbacks.remove(callback)

    def __contains__(self, callback: BaseToolCallbacks) -> bool:
        """Check whether a callback is registered.

        Parameters
        ----------
        callback : BaseToolCallbacks
            The callback to look for.

        Returns
        -------
        bool
            True if the callback is registered, False otherwise.
        """
        return callback in self.callbacks

    def __len__(self) -> int:
        """Return the number of registered callbacks.

        Returns
        -------
        int
            The number of callbacks.
        """
        return len(self.callbacks)

    def __getitem__(self, index: int) -> BaseToolCallbacks:
        """Retrieve a callback by index.

        Parameters
        ----------
        index : int
            The index of the callback to retrieve.

        Returns
        -------
        BaseToolCallbacks
            The callback at the given index.
        """
        return self.callbacks[index]

    def __setitem__(self, index: int, callback: BaseToolCallbacks) -> None:
        """Replace a callback at the given index.

        Parameters
        ----------
        index : int
            The index at which to set the callback.
        callback : BaseToolCallbacks
            The callback to place at the given index.
        """
        self.callbacks[index] = callback

    def __delitem__(self, index: int) -> None:
        """Delete a callback by index.

        Parameters
        ----------
        index : int
            The index of the callback to delete.
        """
        del self.callbacks[index]

    def __iter__(self):
        """Iterate over the registered callbacks.

        Returns
        -------
        iterator
            An iterator over the ``BaseToolCallbacks`` instances.
        """
        return iter(self.callbacks)

    def on_tool_start(self, tool: "BaseTool", *args, **kwargs) -> None:
        """Dispatch the tool-start event to all registered callbacks.

        Parameters
        ----------
        tool : BaseTool
            The tool that is starting execution.
        *args
            Positional arguments passed to the tool.
        **kwargs
            Keyword arguments passed to the tool.
        """
        for callback in self.callbacks:
            callback.on_tool_start(tool, *args, **kwargs)

    def on_tool_activity(self, tool: "BaseTool", activity_type: str, **kwargs) -> Any:
        """Dispatch the tool-activity event to all registered callbacks.

        Parameters
        ----------
        tool : BaseTool
            The tool reporting activity.
        activity_type : str
            A label describing the activity.
        **kwargs
            Additional context about the activity.
        """
        for callback in self.callbacks:
            callback.on_tool_activity(tool, activity_type, **kwargs)

    def on_tool_end(self, tool: "BaseTool", result: Any) -> None:
        """Dispatch the tool-end event to all registered callbacks.

        Parameters
        ----------
        tool : BaseTool
            The tool that finished execution.
        result : Any
            The result produced by the tool.
        """
        for callback in self.callbacks:
            callback.on_tool_end(tool, result)

    def on_tool_error(self, tool: "BaseTool", exception: Exception) -> None:
        """Dispatch the tool-error event to all registered callbacks.

        Parameters
        ----------
        tool : BaseTool
            The tool that encountered an error.
        exception : Exception
            The exception that was raised.
        """
        for callback in self.callbacks:
            callback.on_tool_error(tool, exception)
