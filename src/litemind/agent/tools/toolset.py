from typing import Callable, List, Optional

from litemind.agent.tools.base_tool import BaseTool


class ToolSet:
    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.tools: List[BaseTool] = tools if tools else []

    def add_tool(self, tool: BaseTool):
        """
        Add a new tool.

        Parameters
        ----------
        tool : BaseTool
            The tool to add.

        """
        self.tools.append(tool)

    def add_function_tool(
        self, func: Callable, description: Optional[str] = None
    ) -> "FunctionTool":
        """
        Add a new tool by wrapping a function and its description.

        Parameters
        ----------
        func : Callable
            The function to wrap.
        description : str
            A description of the tool. If no description is provided, the docstring of the function is used instead.
            You can specify which part of the docstring will be used as description by surrounding the description with '***' (e.g. '***This is the description***').

        Returns
        -------
        FunctionTool
            The function tool.

        """
        # Import here to avoid circular imports
        from litemind.agent.tools.function_tool import FunctionTool

        # Create the function tool:
        function_tool = FunctionTool(func, description)

        # Add the tool to the tool set:
        self.tools.append(function_tool)

        # Return the function tool:
        return function_tool

    def add_agent_tool(self, agent, description: str) -> "AgentTool":
        """
        Add a new tool by wrapping an agent and its description.

        Parameters
        ----------
        agent : Agent
            The agent to wrap.
        description : str
            The description of the tool.

        Returns
        -------
        AgentTool
            The tool agent.

        """
        # Import here to avoid circular imports
        from litemind.agent.tools.agent_tool import AgentTool

        # Create the tool agent:
        tool_agent = AgentTool(agent, description)

        # Add the tool to the tool set:
        self.tools.append(tool_agent)

        # Return the tool agent:
        return tool_agent

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        Retrieve a tool by its name.

        Parameters
        ----------

        name : str
            The name of the tool to retrieve.

        Returns
        -------
        Optional[BaseTool]
            The tool, if found, or None.
        """

        # Iterate over tools and return the one with the matching name:
        for tool in self.tools:
            # If the tool name matches, return the tool:
            if tool.name == name:
                # Return the tool:
                return tool

        # If no tool is found, return None:
        return None

    def list_tools(self) -> List[BaseTool]:
        """
        Return all tools as a list.

        Returns
        -------
        List[BaseTool]
            All tools.
        """

        # Return all tools:
        return self.tools

    def tool_names(self) -> List[str]:
        """
        Return all tool names as a list.

        Returns
        -------
        List[str]
            All tool names.
        """

        # Return all tool names:
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
