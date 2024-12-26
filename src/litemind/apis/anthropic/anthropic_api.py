import base64
import os
from typing import List, Optional, Dict, Any

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.openai.exceptions import APIError


class AnthropicApi(BaseApi):

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Anthropic API client.

        Parameters:
        - api_key (Optional[str]): The API key for Anthropic. Defaults to the value of the ANTHROPIC_API_KEY environment variable.
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if api_key is None:
            raise APIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the ANTHROPIC_API_KEY environment variable"
            )

        # Initialize the Anthropic client
        import anthropic
        self.client = anthropic.Client(api_key=api_key)

    def check_api_key(self, api_key: Optional[str] = None) -> bool:
        """
        Test the API key with a simple completion.

        Returns:
        - bool: True if the API key is valid, False otherwise.
        """
        try:
            self.completion([Message(role="user", content="Hello!")])
            return True
        except Exception:
            return False

    def has_vision_support(self) -> bool:
        """
        Indicates whether the API supports vision tasks.

        Returns:
        - bool: True (Claude 3 and above support vision).
        """
        return True

    def max_num_input_token(self, model: str) -> int:
        """
        Returns the maximum number of input tokens allowed for a specific model.

        Parameters:
        - model (str): The model name.

        Returns:
        - int: Maximum token limit.
        """
        if "claude-3" in model:
            return 100000
        elif "claude-2" in model:
            return 100000
        elif "claude-1" in model:
            return 9000
        else:
            raise ValueError(f"Unknown model: {model}")

    def completion(self,
                   messages: List[Message],
                   model_name: Optional[str] = "claude-3",
                   temperature: Optional[float] = 0.0,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:
        """
        Generate a completion using the Anthropic API.

        Parameters:
        - messages (List[Message]): A list of messages to send to the API.
        - model (Optional[str]): The model to use for the request.
        - temperature (Optional[float]): Sampling temperature.
        - toolset (Optional[ToolSet]): A set of tools for Claude to use during the interaction.

        Returns:
        - Message: The response from Claude.
        """
        # Convert messages into Anthropic's format
        anthropic_messages = self._convert_messages_for_anthropic(messages)

        # Prepare the tool definitions if a ToolSet is provided
        tools = self._convert_toolset_to_anthropic(toolset) if toolset else []

        # Call the API
        try:
            response = self.client.completions.create(
                model=model_name,
                temperature=temperature,
                tools=tools,
                messages=anthropic_messages,
                **kwargs
            )

            # Handle tool use or standard completions
            if response.get("stop_reason") == "tool_use":
                tool_name = response["content"][0]["name"]
                tool_input = response["content"][0]["input"]
                return Message(
                    role="assistant",
                    content={"tool_name": tool_name, "tool_input": tool_input}
                )

            content = response["completion"]
            response_message = Message(role="assistant", content=content)
            messages.append(response_message)
            return response_message

        except Exception as e:
            raise APIError(f"Error during completion: {e}")

    def describe_image(self,
                       image_path: str,
                       query: str = 'Here is an image, please carefully describe it in detail.',
                       model_name: str = "claude-3-5-sonnet",
                       max_tokens: int = 1024,
                       temperature: float = 0.0) -> str:
        """
        Describe an image using Claude's vision capabilities.

        Parameters:
        - image_path (str): Path to the image to describe.
        - query (str): Query to send to Claude for analyzing the image.
        - model (str): Claude model to use for image analysis.
        - max_tokens (int): Maximum number of tokens for the response.
        - temperature (float): Sampling temperature.

        Returns:
        - str: Claude's response describing the image.
        """
        valid_formats = {'.png': 'image/png', '.jpg': 'image/jpeg',
                         '.jpeg': 'image/jpeg', '.gif': 'image/gif',
                         '.webp': 'image/webp'}
        file_extension = os.path.splitext(image_path)[-1].lower()
        if file_extension not in valid_formats:
            raise ValueError(
                f"Unsupported image format: {file_extension}. Supported formats are {list(valid_formats.keys())}.")

        with open(image_path, "rb") as image_file:
            encoded_image = base64.standard_b64encode(image_file.read()).decode(
                "utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": valid_formats[file_extension],
                            "data": encoded_image,
                        },
                    },
                    {
                        "type": "text",
                        "text": query,
                    },
                ],
            }
        ]

        try:
            response = self.client.messages.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
            return response["completion"]
        except Exception as e:
            raise APIError(f"Error during image description: {e}")

    @staticmethod
    def _convert_messages_for_anthropic(messages: List[Message]) -> List[dict]:
        """
        Convert messages into Anthropic's expected format.

        Parameters:
        - messages (List[Message]): A list of messages.

        Returns:
        - List[dict]: Converted messages.
        """
        anthropic_messages = []
        for msg in messages:
            anthropic_messages.append({
                "role": msg.role,
                "content": msg.content
            })
        return anthropic_messages

    @staticmethod
    def _convert_toolset_to_anthropic(toolset: ToolSet) -> List[Dict[str, Any]]:
        """
        Convert a ToolSet into a format compatible with Anthropic's API.

        Parameters:
        - toolset (ToolSet): The ToolSet object containing tools.

        Returns:
        - List[Dict[str, Any]]: List of tool definitions.
        """
        tools = []
        for tool in toolset.list_tools():
            tool_definition = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema if hasattr(tool,
                                                             "input_schema") else {}
            }
            tools.append(tool_definition)
        return tools
