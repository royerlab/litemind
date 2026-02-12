"""Managed collection of tools available to an Agent.

This module provides the ``ToolSet`` class, a list-like container that
holds ``BaseTool`` instances and offers convenience methods for adding
function tools, agent tools, and built-in tools, as well as managing
callbacks across all contained tools.
"""

from typing import Callable, List, Optional, Type, Union

from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.callbacks.base_tool_callbacks import BaseToolCallbacks


class ToolSet:
    """A managed collection of tools available to an Agent.

    Provides methods to add, remove, and query tools of various types
    (function tools, agent tools, built-in tools) and to manage tool
    callbacks across all contained tools.
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

    def remove_tool(self, tool: Union[str, BaseTool]) -> bool:
        """
        Remove a tool by name or instance.

        Parameters
        ----------
        tool : Union[str, BaseTool]
            The tool name (str) or tool instance (BaseTool) to remove.

        Returns
        -------
        bool
            True if the tool was removed, False if not found.
        """
        if isinstance(tool, str):
            # Find tool by name
            for t in self.tools:
                if t.name == tool:
                    self.tools.remove(t)
                    return True
            return False
        else:
            # Remove by instance
            if tool in self.tools:
                self.tools.remove(tool)
                return True
            return False

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
        """Retrieve a tool by index.

        Parameters
        ----------
        item : int
            The index of the tool to retrieve.

        Returns
        -------
        BaseTool
            The tool at the given index.
        """
        return self.tools[item]

    def __len__(self) -> int:
        """Return the number of tools in the set.

        Returns
        -------
        int
            The number of tools.
        """
        return len(self.tools)

    def __iter__(self):
        """Iterate over the tools in the set.

        Returns
        -------
        iterator
            An iterator over the ``BaseTool`` instances.
        """
        return iter(self.tools)

    def __contains__(self, item) -> bool:
        """Check whether a tool instance is in the set.

        Parameters
        ----------
        item : BaseTool
            The tool instance to look for.

        Returns
        -------
        bool
            True if the tool is present, False otherwise.
        """
        return item in self.tools

    def __delitem__(self, key):
        """Delete a tool by index.

        Parameters
        ----------
        key : int
            The index of the tool to delete.
        """
        del self.tools[key]

    def __setitem__(self, key, value):
        """Replace a tool at the given index.

        Parameters
        ----------
        key : int
            The index at which to set the tool.
        value : BaseTool
            The tool to place at the given index.
        """
        self.tools[key] = value

    def __iadd__(self, other: BaseTool):
        """Add a tool in place using the ``+=`` operator.

        Parameters
        ----------
        other : BaseTool
            The tool to add.

        Returns
        -------
        ToolSet
            This tool set (modified in place).
        """
        self.add_tool(other)
        return self

    def __add__(self, other: BaseTool) -> "ToolSet":
        """Create a new ToolSet with an additional tool using the ``+`` operator.

        Parameters
        ----------
        other : BaseTool
            The tool to append.

        Returns
        -------
        ToolSet
            A new ``ToolSet`` containing all existing tools plus ``other``.
        """
        new_toolset = ToolSet(tools=list(self.tools))
        new_toolset.add_tool(other)
        return new_toolset

    def __str__(self):
        """Return a concise string listing tool names.

        Returns
        -------
        str
            A string of the form ``ToolSet([name1, name2, ...])``.
        """
        return f"ToolSet({[t.name for t in self.tools]})"

    def __repr__(self):
        """Return a detailed string representation including full tool details.

        Returns
        -------
        str
            A string of the form ``ToolSet([tool1_repr, tool2_repr, ...])``.
        """
        return f"ToolSet({self.tools})"
