from typing import Optional, Any, Dict

from litemind.agent.agent import Agent
from litemind.agent.tools.base_tool import BaseTool


class AgentTool(BaseTool):
    """Tool that wraps an agent, calling it with a prompt and returning structured output."""

    def __init__(self,
                 agent: Agent,
                 description: str,
                 has_memory: Optional[bool] = False,
                 ):

        if not isinstance(agent, Agent):
            raise ValueError("AgentTool must be initialized with an Agent object")

        if not agent.name or not description:
            raise ValueError("AgentTool must be initialized with a name and description")

        super().__init__(agent.name, description)
        self.agent = agent
        self.has_memory = has_memory

    def execute(self, prompt: str) -> Any:
        """Execute the agent with a prompt and return a structured output."""

        # Clear the conversation if the agent does not have memory:
        if not self.has_memory:
            self.agent.conversation.clear_conversation()

        # Call the agent with the prompt:
        response_message = self.agent(text=prompt)

        # Return the response text:
        response_text = response_message.text if hasattr(response_message, 'text') else str(response_message)
        return response_text