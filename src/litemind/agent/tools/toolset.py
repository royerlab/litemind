from typing import List, Optional, Callable


from litemind.agent.tools.base_tool import BaseTool



class ToolSet:
    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.tools : List[BaseTool] = tools if tools else []

    def add_tool(self, tool: BaseTool):
        """Add a new tool."""
        self.tools.append(tool)

    def add_function_tool(self, func: Callable, description: str):
        """Add a new tool by wrapping a function and its description."""
        from litemind.agent.tools.function_tool import FunctionTool
        self.tools.append(FunctionTool(func, description))

    def add_agent_tool(self, agent, description: str):
        """Add a new tool by wrapping an agent and its description."""
        from litemind.agent.tools.agent_tool import AgentTool
        self.tools.append(AgentTool(agent, description))

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by its name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def list_tools(self) -> List[BaseTool]:
        """Return all tools as a list."""
        return self.tools
