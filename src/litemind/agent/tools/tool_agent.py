from typing import Any, Optional

from arbol import asection

from litemind.agent.agent import Agent
from litemind.agent.tools.base_tool import BaseTool


class ToolAgent(BaseTool):
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
        self.parameters = {
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
            response_message = self.agent(text=prompt)

            # Return the response text:
            response_text = (
                response_message.text
                if hasattr(response_message, "text")
                else str(response_message)
            )
            return response_text
