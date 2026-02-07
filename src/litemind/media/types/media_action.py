from litemind.agent.messages.actions.action_base import ActionBase
from litemind.media.media_default import MediaDefault


class Action(MediaDefault):
    """Media that represents an action such as a tool call or result.

    Used internally by the agent system to embed actions (tool calls,
    tool results, etc.) within the message stream.
    """

    def __init__(self, action: ActionBase, **kwargs):
        """Create a new action media.

        Parameters
        ----------
        action : ActionBase
            The action instance to wrap.
        **kwargs
            Additional keyword arguments forwarded to ``MediaDefault``.

        Raises
        ------
        ValueError
            If *action* is None or not an ``ActionBase`` instance.
        """

        super().__init__(**kwargs)

        # Validate Object here:
        if action is None:
            raise ValueError("Object cannot be None")
        elif not isinstance(action, ActionBase):
            raise ValueError(f"Object must be an ActionBase, got {type(action)}")

        # Set attributes:
        self.action: ActionBase = action

    def get_content(self) -> ActionBase:
        """Return the wrapped action instance.

        Returns
        -------
        ActionBase
            The stored action.
        """
        return self.action
