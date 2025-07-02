import inspect
from typing import Any, Callable, Dict, Optional

from arbol import aprint, asection

from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.utils.inspect_function import extract_docstring


class FunctionTool(BaseTool):
    def __init__(self, func: Callable, description: Optional[str] = None):
        """
        Initialize a tool that wraps a function.

        Parameters
        ----------
        func:
            The function to wrap.
        description:
            A description of the tool. If no description is provided, the docstring of the function is used instead.
            You can specify which part of the docstring will be used as description by surrounding the description with '***' (e.g. '***This is the description***').

        """
        # Initialize the base tool, description is empty for now...
        super().__init__(name=func.__name__, description="")

        # Store the function
        self.func = func

        # Use the provided description or extract it from the function's docstring
        if not description:
            docstring = extract_docstring(func)
            # if '***' is present, extract the substring between the first and second occurrence of '***':
            if "***" in docstring:
                self.description = docstring[
                    docstring.find("***")
                    + 3 : docstring.find("***", docstring.find("***") + 1)
                ]
            else:
                self.description = docstring
        else:
            self.description = description
        self.name = func.__name__
        self.arguments_schema, self.arg_and_type = self._generate_arguments_schema()

    def _generate_arguments_schema(self) -> Dict[str, Any]:
        """Generate a JSON schema for the function parameters based on type hints."""

        arg_and_type = {}

        schema = {"type": "object", "properties": {}, "required": []}
        sig = inspect.signature(self.func)

        for name, param in sig.parameters.items():

            # Get the type of the parameter
            param_type = param.annotation

            # Handle untyped parameters as optional
            if param_type == inspect._empty:
                param_type = str  # Default to string type if no type is specified

            # Get the type string
            type_str = self._map_type_to_json_schema(param_type)

            # Add the parameter to the schema
            schema["properties"][name] = {"type": type_str}

            if param.default == inspect._empty:
                schema["required"].append(name)

            # Store the type of the parameter
            arg_and_type[name] = type_str

        schema["additionalProperties"] = False
        return schema, arg_and_type

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

    def _execute(self, *args, **kwargs) -> Any:
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
        with asection(f"Executing tool '{self.name}'"):
            try:
                aprint(f"Arguments: {args}, {kwargs}")
                result = self.func(*args, **kwargs)
                aprint(f"Result: {result}")
            except Exception as e:
                # print the stacktrace:
                import traceback

                traceback.print_exc()
                aprint(f"Error: {e}")
                raise e

        return result

    def pretty_string(self):
        """
        Return a pretty string representation of the tool.

        Returns
        -------
        str
            A pretty string representation of the tool.
        """
        # Make a pretty string representation of the arg_and_type dict:
        arguments = ", ".join([f"{k}: {v}" for k, v in self.arg_and_type.items()])

        # Shorten description to the first period _after_ 80 characters:
        if len(self.description) > 80:
            description = (
                self.description[: self.description.find(".", 80) + 1] + "[...]"
            )
        else:
            description = self.description

        return f"{self.name}({arguments}) % {description}"
