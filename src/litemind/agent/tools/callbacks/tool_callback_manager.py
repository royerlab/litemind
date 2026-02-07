from typing import Any, List, Sequence, Union

from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks


class ToolCallbackManager(BaseToolCallbacks):
    """Manages a collection of tool callbacks and dispatches lifecycle events.

    Aggregates multiple ``BaseToolCallbacks`` instances and fans out
    ``on_tool_start``, ``on_tool_end``, ``on_tool_activity``, and
    ``on_tool_error`` calls to all registered callbacks.
    """

    def __init__(self):
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
        return callback in self.callbacks

    def __len__(self) -> int:
        return len(self.callbacks)

    def __getitem__(self, index: int) -> BaseToolCallbacks:
        return self.callbacks[index]

    def __setitem__(self, index: int, callback: BaseToolCallbacks) -> None:
        self.callbacks[index] = callback

    def __delitem__(self, index: int) -> None:
        del self.callbacks[index]

    def __iter__(self):
        return iter(self.callbacks)

    def on_tool_start(self, tool: "BaseTool", *args, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_tool_start(tool, *args, **kwargs)

    def on_tool_activity(self, tool: "BaseTool", activity_type: str, **kwargs) -> Any:
        for callback in self.callbacks:
            callback.on_tool_activity(tool, activity_type, **kwargs)

    def on_tool_end(self, tool: "BaseTool", result: Any) -> None:
        for callback in self.callbacks:
            callback.on_tool_end(tool, result)

    def on_tool_error(self, tool: "BaseTool", exception: Exception) -> None:
        for callback in self.callbacks:
            callback.on_tool_error(tool, exception)
