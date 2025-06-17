from typing import Any, List

from litemind.agent.tools.builtin_tools.builtin_tool import BuiltinTool


class BuiltinWebSearchTool(BuiltinTool):
    """A tool that represents built-in web search functionality."""

    def __init__(
        self,
        search_context_size: str = "medium",
        max_web_searches: int = 10,
        allowed_domains: List[str] = None,
    ):
        """
        Initialize a built-in web search tool. This is used to carry the parameters for built-in web search functionality.
        Different APIs and models might have different parameters.


        Parameters
        ----------
        search_context_size: str
            The size of the search context to use. This can be 'high', 'medium', or 'low'. Defaults to 'medium'.
        max_web_searches: int
            The maximum number of web searches to perform. Defaults to 10.
        allowed_domains: List[str]
            A list of allowed domains to restrict the search to. If None, no restrictions are applied.

        """
        # Initialize the base tool, description is empty for now...
        super().__init__(
            name=BuiltinWebSearchTool.__name__, description="Built-in web search tool"
        )

        # Validate the search context size:
        if search_context_size not in ["high", "medium", "low"]:
            raise ValueError(
                "search_context_size must be one of 'high', 'medium', or 'low'."
            )
        # Set the search context size:
        self.search_context_size = search_context_size

        # Set the maximum number of web searches:
        if not isinstance(max_web_searches, int) or max_web_searches <= 0:
            raise ValueError("max_web_searches must be a positive integer.")
        self.max_web_searches = max_web_searches

        if allowed_domains is None:
            allowed_domains = []
        # Set the allowed domains, if any:
        if not isinstance(allowed_domains, list):
            raise ValueError("allowed_domains must be a list of strings.")
        else:
            self.allowed_domains = (
                allowed_domains if allowed_domains is not None else []
            )

    def execute(self, *args, **kwargs) -> Any:
        raise RuntimeError(
            f"{BuiltinWebSearchTool.__name__} tool cannot be executed directly. It is a placeholder for built-in web search functionality."
        )

    def pretty_string(self):
        """
        Return a pretty string representation of the tool.

        Returns
        -------
        str
            A pretty string representation of the tool.
        """
        return f"{self.name} (Built-in web search tool, cannot be executed directly)"
