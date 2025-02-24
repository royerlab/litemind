from typing import List, Optional, Callable

from litemind.agent.tools.base_tool import BaseTool


class ToolSet:
    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.tools: List[BaseTool] = tools if tools else []

    def add_tool(self, tool: BaseTool):
        """Add a new tool."""
        self.tools.append(tool)

    def add_function_tool(self, func: Callable, description: str):
        """Add a new tool by wrapping a function and its description."""
        # Import here to avoid circular imports
        from litemind.agent.tools.function_tool import FunctionTool

        self.tools.append(FunctionTool(func, description))

    def add_agent_tool(self, agent, description: str):
        """Add a new tool by wrapping an agent and its description."""
        # Import here to avoid circular imports
        from litemind.agent.tools.tool_agent import ToolAgent

        self.tools.append(ToolAgent(agent, description))

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Retrieve a tool by its name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def list_tools(self) -> List[BaseTool]:
        """Return all tools as a list."""
        return self.tools

    def tool_names(self) -> List[str]:
        """Return all tool names as a list."""
        return [t.name for t in self.tools]

    def __getitem__(self, item) -> BaseTool:
        return self.tools[item]

    def __len__(self) -> int:
        return len(self.tools)

    def __iter__(self):
        return iter(self.tools)

    def __contains__(self, item) -> bool:
        return item in self.tools

    def __delitem__(self, key):
        del self.tools[key]

    def __setitem__(self, key, value):
        self.tools[key] = value

    def __iadd__(self, other: BaseTool):
        self.add_tool(other)
        return self

    def __add__(self, other: BaseTool):
        self.add_tool(other)
        return self

    def __str__(self):
        return f"ToolSet({[t.name for t in self.tools]})"

    def __repr__(self):
        # provide all information available about tools:
        return f"ToolSet({self.tools})"
