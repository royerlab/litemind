from litemind.agent.tools.base_tool import BaseTool


class BuiltinTool(BaseTool):
    """
    A tool that represents 'built-in' tools.
    These are tools that are provided by the API or model itself, such as web search, Python, image generation, etc.
    This class is used to identify built-in tools and to provide a common interface for them if
    """
