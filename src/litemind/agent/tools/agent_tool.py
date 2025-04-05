from typing import Any, Optional

from arbol import asection

from litemind.agent.agent import Agent
from litemind.agent.tools.base_tool import BaseTool


class AgentTool(BaseTool):
    """Tool that wraps an agent, calling it with a prompt and returning structured output."""

    def __init__(
        self,
        agent: Agent,
        description: str,
        has_memory: Optional[bool] = False,
    ):
        """
        Create a new tool agent.

        Parameters
        ----------
        agent: Agent
            The agent to wrap.
        description: str
            The description of the tool.
        has_memory: bool
            If True, the agent will keep the conversation history between calls
        """

        if not isinstance(agent, Agent):
            raise ValueError("AgentTool must be initialized with an Agent object")

        if not agent.name or not description:
            raise ValueError(
                "AgentTool must be initialized with a name and description"
            )

        # Initialize the tool:
        super().__init__(agent.name, description)

        self.agent = agent
        self.has_memory = has_memory

        # Set the parameters to correspond to a function that takes a prompt (string) and returns a string:
        self.arguments_schema = {
            "type": "object",
            "properties": {"prompt": {"type": "string"}},
            "required": ["prompt"],
            "additionalProperties": False,
        }

    def execute(self, prompt: str) -> Any:
        """Execute the agent with a prompt and return a structured output."""
        with asection(f"Executing tool agent '{self.name}'"):
            # Clear the conversation if the agent does not have memory:
            if not self.has_memory:
                self.agent.conversation.clear_conversation()

            # Call the agent with the prompt:
            response = self.agent(prompt)

            # Get the last message of the response:
            last_response_message = response[-1]

            # Return the response text:
            response_text = last_response_message.to_plain_text()
            return response_text

    def pretty_string(self):
        """
        Return a pretty string representation of the tool agent.

        Returns
        -------
        str
            A pretty string representation of the tool agent.
        """

        # Shorten description to the first period _after_ 80 characters:
        if len(self.description) > 80:
            description = (
                self.description[: self.description.find(".", 80) + 1] + "[...]"
            )
        else:
            description = self.description

        return f"{self.agent.name}(prompt: str) -> str  % {description}"
