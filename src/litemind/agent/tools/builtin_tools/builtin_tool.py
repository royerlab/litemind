"""Base class for built-in tools provided natively by an API or model.

This module defines ``BuiltinTool``, a marker subclass of ``BaseTool``
for tools whose execution is handled by the API provider rather than
locally (e.g., web search, code execution, image generation).
"""

from litemind.agent.tools.base_tool import BaseTool


class BuiltinTool(BaseTool):
    """Base class for tools provided natively by an API or model.

    Built-in tools (e.g., web search, code execution, image generation)
    are not executed locally but are handled by the API provider. This
    class serves as a marker and common interface for such tools.
    """
