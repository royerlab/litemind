from typing import Any, Dict, List, Optional

from litemind.agent.tools.builtin_tools.builtin_tool import BuiltinTool


class BuiltinMCPTool(BuiltinTool):
    """A tool that represents built-in mcp tools."""

    def __init__(
        self,
        server_name: str,
        server_url: str,
        headers: Optional[Dict[str, str]] = None,
        allowed_tools: Optional[List[str]] = None,
    ):
        """
        Initialize a built-in mcp tool given a server name, URL, and optionally headers and allowed tools.

        Parameters
        ----------
        server_name: str
            The name of the MCP server.
        server_url: str
            The URL of the MCP server.
        headers: Optional[Dict[str, str]]
            Optional headers to include in requests to the MCP server -- for example, authentication headers.
        allowed_tools: Optional[List[str]]
            A list of allowed tools that can be used with this MCP server. If None, all tools are allowed.


        """
        # Initialize the base tool, description is empty for now...
        super().__init__(
            name=f"{server_name} MCP server tool",
            description="Built-in MCP server tool",
        )

        # Set the server name and URL:
        self.server_name = server_name
        self.server_url = server_url
        self.headers = headers if headers is not None else {}
        self.allowed_tools = allowed_tools if allowed_tools is not None else []

    def execute(self, *args, **kwargs) -> Any:
        raise RuntimeError(
            f"{BuiltinMCPTool.__name__} tool cannot be executed directly. It is a placeholder for built-in MCP functionality."
        )

    def pretty_string(self):
        """
        Return a pretty string representation of the tool.

        Returns
        -------
        str
            A pretty string representation of the tool.
        """
        return f"{self.name} (Built-in MCP tool, cannot be executed directly)"
