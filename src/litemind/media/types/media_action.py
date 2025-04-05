from litemind.agent.messages.actions.action_base import ActionBase
from litemind.media.media_default import MediaDefault


class Action(MediaDefault):
    """
    An Action is a media that describes an 'action' such as a tool call or use, or any other such action.
    """

    def __init__(self, action: ActionBase, **kwargs):
        """
        Create a new object information.

        Parameters
        ----------
        action: ActionBase
            Action

        """

        super().__init__(**kwargs)

        # Validate Object here:
        if action is None:
            raise ValueError(f"Object cannot be None")
        elif not isinstance(action, ActionBase):
            raise ValueError(f"Object must be an ActionBase, got {type(action)}")

        # Set attributes:
        self.action: ActionBase = action

    def get_content(self) -> ActionBase:
        return self.action
