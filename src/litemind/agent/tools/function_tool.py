import inspect
from typing import Callable, Dict, Any

from litemind.agent.tools.base_tool import BaseTool


class FunctionTool(BaseTool):
    def __init__(self, func: Callable, description: str):
        self.func = func
        self.description = description
        self.name = func.__name__
        self.parameters = self._generate_parameters_schema()

    def _generate_parameters_schema(self) -> Dict[str, Any]:
        """Generate a JSON schema for the function parameters based on type hints."""
        schema = {"type": "object", "properties": {}, "required": []}
        sig = inspect.signature(self.func)

        for name, param in sig.parameters.items():
            param_type = param.annotation
            # Handle untyped parameters as optional
            if param_type == inspect._empty:
                param_type = str  # Default to string type if no type is specified
            type_str = self._map_type_to_json_schema(param_type)
            schema["properties"][name] = {"type": type_str}
            if param.default == inspect._empty:
                schema["required"].append(name)

        schema["additionalProperties"] = False
        return schema

    def _map_type_to_json_schema(self, py_type: Any) -> str:
        """Map Python types to JSON schema types."""
        if py_type in {int, float}:
            return "number"
        elif py_type == bool:
            return "boolean"
        elif py_type == list:
            return "array"
        elif py_type == dict:
            return "object"
        else:
            return "string"

    def execute(self, *args, **kwargs) -> Any:
        """
        Execute the tool with given arguments.

        Parameters
        ----------
        *args
            Positional arguments to pass to the tool

        **kwargs
            Arbitrary keyword arguments to pass to the tool function.

        Returns
        -------
        Any
            The result of the tool function.
        """
        return self.func(*args, **kwargs)
