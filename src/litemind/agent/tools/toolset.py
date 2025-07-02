from typing import Callable, List, Optional, Type, Union

from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks


class ToolSet:
    """
    A set of tools that can be used by an agent.
    This class allows you to manage a collection of tools, add new tools, and interact with them.
    It provides methods to add function tools, agent tools, built-in tools, and manage tool callbacks.
    It also provides methods to check for the presence of tools, retrieve tools by name, and list all tools.
    """

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        """
        Create a new ToolSet.
        This initializes the ToolSet with an optional list of tools.
        If no tools are provided, an empty ToolSet is created.

        Parameters
        ----------
        tools: Optional[List[BaseTool]]
            A list of tools to initialize the ToolSet with. If None, an empty list is used.
        """
        self.tools: List[BaseTool] = tools if tools else []

    def add_tool_callback(self, tool_callback: BaseToolCallbacks):
        """
        Add a tool callback to all tools in the ToolSet.

        Parameters
        ----------
        tool_callback : BaseToolCallbacks
            The tool callback to add to all tools present in the ToolSet.

        """
        # Add the tool callback to each tool:
        for tool in self.tools:
            tool.callbacks.add_callback(tool_callback)

    def remove_tool_callback(self, tool_callback: BaseToolCallbacks):
        """
        Remove a tool callback from all tools in the ToolSet.

        Parameters
        ----------
        tool_callback : BaseToolCallbacks
            The tool callback to remove from all tools present in the ToolSet.

        """
        # Remove the tool callback from each tool:
        for tool in self.tools:
            tool.callbacks.remove_callback(tool_callback)

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

    def add_builtin_web_search_tool(self) -> "BuiltinWebSearchTool":
        """
        Add a built-in web search tool.

        Returns
        -------
        BuiltinWebSearchTool
            The builtin web search tool.
        """

        # Create the web search tool:
        from litemind.agent.tools.builtin_tools.web_search_tool import (
            BuiltinWebSearchTool,
        )

        builtin_web_search_tool = BuiltinWebSearchTool()

        # Add the tool to the tool set:
        self.tools.append(builtin_web_search_tool)

        # Return the web search tool:
        return builtin_web_search_tool

    def add_builtin_mcp_tool(
        self,
        server_name: str,
        server_url: str,
        headers: Optional[dict] = None,
        allowed_tools: Optional[List[str]] = None,
    ) -> "BuiltinMCPTool":
        """
        Add a built-in MCP tool.

        Parameters
        ----------
        server_name : str
            The name of the MCP server.
        server_url : str
            The URL of the MCP server.
        headers : Optional[dict]
            Optional headers to include in requests to the MCP server.
        allowed_tools : Optional[List[str]]
            A list of allowed tools that can be used with this MCP server. If None, all tools are allowed.

        Returns
        -------
        BuiltinMCPTool
            The builtin MCP tool.
        """

        # Create the MCP tool:
        from litemind.agent.tools.builtin_tools.mcp_tool import BuiltinMCPTool

        builtin_mcp_tool = BuiltinMCPTool(
            server_name, server_url, headers, allowed_tools
        )

        # Add the tool to the tool set:
        self.tools.append(builtin_mcp_tool)

        # Return the MCP tool:
        return builtin_mcp_tool

    def remove_tool(self, tool: BaseTool):
        """
        Remove a tool.

        Parameters
        ----------
        tool : BaseTool
            The tool to remove.

        """
        # Remove the tool from the tools list:
        self.tools.remove(tool)

    def has_tool(self, tool: Union[str, BaseTool, Type[BaseTool]]) -> bool:
        """
        Check if a tool is present in the tool set.

        Parameters
        ----------
        tool : Union[str, BaseTool, Type[BaseTool]]
            The tool to check for. Can be a string (tool name), a BaseTool instance, or a type of BaseTool.

        Returns
        -------
        bool
            True if the tool is present, False otherwise.
        """

        # If the tool is a string, check by name:
        if isinstance(tool, str):
            return any(t.name == tool for t in self.tools)

        # If the tool is a BaseTool instance, check by instance:
        elif isinstance(tool, BaseTool):
            return tool in self.tools

        # If the tool is a type of BaseTool, check by type:
        elif isinstance(tool, type) and issubclass(tool, BaseTool):
            return any(isinstance(t, tool) for t in self.tools)

        # If none of the above, return False:
        return False

    def get_tool(self, name: Union[str, Type[BaseTool]]) -> Optional[BaseTool]:
        """
        Retrieve a tool by its name.

        Parameters
        ----------

        name : Union[str, Type[BaseTool]]
            The name of the tool to retrieve, or the type (class) of the tool.

        Returns
        -------
        Optional[BaseTool]
            The tool, if found, or None.
        """

        # if the name is a type, return the first tool of that type:
        if isinstance(name, type) and issubclass(name, BaseTool):
            # Get the name of the tool class:
            name = name.__name__

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

    def list_builtin_tools(self) -> List[BaseTool]:
        """
        Return all built-in tools as a list.

        Returns
        -------
        List[BaseTool]
            All built-in tools.
        """

        # Filter and return only built-in tools:
        return [tool for tool in self.tools if tool.is_builtin()]

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
