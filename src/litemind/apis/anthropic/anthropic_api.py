import os
from typing import List, Optional

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.anthropic.utils.messages import \
    _convert_messages_for_anthropic
from litemind.apis.anthropic.utils.process_response import _process_response
from litemind.apis.anthropic.utils.tools import _convert_toolset_to_anthropic
from litemind.apis.base_api import BaseApi
from litemind.apis.exceptions import APIError


class AnthropicApi(BaseApi):
    """
    An Anthropic API implementation that conforms to the `BaseApi` abstract interface.
    Uses the `anthropic` library's `client.messages.create(...)` methods for completions.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize the Anthropic client.

        Parameters
        ----------
        api_key : Optional[str]
            The API key for Anthropic. If not provided, we'll read from ANTHROPIC_API_KEY env var.
        kwargs : dict
            Additional options (e.g. `timeout=...`, `max_retries=...`) passed to `Anthropic(...)`.
        """
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise APIError(
                "An Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY or pass `api_key=...` explicitly."
            )

        # Create the Anthropic client
        from anthropic import Anthropic
        self.client = Anthropic(
            api_key=api_key,
            **kwargs  # e.g. timeout=30.0, max_retries=2, etc.
        )

    def check_api_key(self, api_key: Optional[str] = None) -> bool:

        # Use the provided key if any, else the one from the client
        candidate_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not candidate_key:
            return False

        # We'll attempt a trivial request
        try:
            test_message = {
                "role": "user",
                "content": "Hello, is this key valid?",
            }
            resp = self.client.messages.create(
                model=self.default_model(),
                max_tokens=16,
                messages=[test_message],
            )
            # If no exception: assume it's valid enough
            _ = resp.content  # Accessing content to ensure it exists
            return True
        except Exception:
            # printout stacktrace:
            import traceback
            traceback.print_exc()
            return False

    def model_list(self) -> List[str]:
        """
        Return a list of known Anthropic models.
        """
        try:
            # List Anthropic models:
            from anthropic.types import ModelInfo
            from openai.pagination import SyncPage

            # Get the first 100 models:
            models_info: List[ModelInfo] = self.client.models.list(
                limit=100).data

            # Extract model IDs
            model_list: List[str] = list([str(info.id) for info in models_info])

            return model_list
        except Exception as e:
            # Handle error and return an empty list
            import traceback
            traceback.print_exc()
            return []

    def default_model(self,
                      require_images: bool = False,
                      require_audio: bool = False,
                      require_tools: bool = False) -> Optional[str]:

        # Get the list of models:
        model_list = self.model_list()

        if require_images:
            # Filter out models that don't support images
            model_list = [m for m in model_list if self.has_image_support(m)]

        if require_audio:
            # Filter out models that don't support audio
            model_list = [m for m in model_list if self.has_audio_support(m)]

        if require_tools:
            # Filter out models that don't support tools
            model_list = [m for m in model_list if self.has_tool_support(m)]

        # If we have any models left, return the first one
        if model_list:
            return model_list[0]
        else:
            return None

    def has_image_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.default_model()

        # For a simple check:
        return "3-5" in model_name or "sonnet" in model_name or "vision" in model_name

    def has_audio_support(self, model_name: Optional[str] = None) -> bool:

        # No Anthropic models currently support Audio.
        return False

    def has_tool_support(self, model_name: Optional[str] = None) -> bool:
        """
        Return True if the model supports function-calling (tools).
        Per Anthropic's samples, Claude 2+ often does.
        We'll guess 'claude-2' or 'claude-3' or 'sonnet' => True.
        """
        if model_name is None:
            model_name = self.default_model()

        return ("claude-2" in model_name or "claude-3" in model_name)

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Return the maximum *input tokens* (context window) for an Anthropic Claude model.

        If model_name is unrecognized, fallback = 100_000 (safe guess).
        """

        if model_name is None:
            model_name = self.default_model()

        name = model_name.lower()

        # Claude 3.5 Sonnet/Haiku => 200k
        if "claude-3-5-sonnet" in name or "claude-3-5-haiku" in name:
            return 200_000

        # Claude 3 (Opus, Sonnet, Haiku) => 200k
        if "claude-3-opus" in name or "claude-3-sonnet" in name or "claude-3-haiku" in name:
            return 200_000

        # Claude 2.1 => 200k
        if "claude-2.1" in name:
            return 200_000

        # Claude 2 => 100k
        if "claude-2.0" in name or "claude-2" in name:
            return 100_000

        # Claude Instant 1.2 => 100k
        if "claude-instant-1.2" in name:
            return 100_000

        # Fallback
        return 100_000

    def max_num_output_tokens(self,
                              model_name: Optional[str] = None) -> int:
        """
        Return the maximum *output tokens* that can be generated by a given Anthropic Claude model.

        If model_name is unrecognized, fallback = 4096.
        """
        if model_name is None:
            model_name = self.default_model()

        name = model_name.lower()

        # Claude 3.5 Sonnet/Haiku => 8192
        if "claude-3-5-sonnet" in name or "claude-3-5-haiku" in name:
            return 8192

        # Claude 3 Opus/Sonnet/Haiku => 4096
        if "claude-3-opus" in name or "claude-3-sonnet" in name or "claude-3-haiku" in name:
            return 4096

        # Claude 2.1 => 4096
        if "claude-2.1" in name:
            return 4096

        # Claude 2 => 4096
        if "claude-2.0" in name or "claude-2" in name:
            return 4096

        # Claude Instant 1.2 => 4096
        if "claude-instant-1.2" in name:
            return 4096

        # Fallback
        return 4096

    def completion(self,
                   model_name: str,
                   messages: List[Message],
                   temperature: float = 0.0,
                   max_output_tokens: Optional[int] = None,
                   toolset: Optional[ToolSet] = None,
                   **kwargs) -> Message:

        from anthropic import NotGiven

        # Extract system message if present
        system_messages = ''
        non_system_messages = []
        for msg in messages:
            if msg.role == "system":
                system_messages += msg.text
            else:
                non_system_messages.append(msg)

        # Convert remaining non-system litemind Messages to Anthropic messages:
        anthropic_messages = _convert_messages_for_anthropic(
            non_system_messages)

        # Convert a ToolSet to Anthropic "tools" param if any
        tools_param = _convert_toolset_to_anthropic(
            toolset) if toolset else NotGiven()

        # Get max num of output tokens for model if not provided:
        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        # We can pass temperature to anthropic via e.g. 'temperature' or 'temperature' param.
        # The param for completions is 'temperature' in the library’s .messages.create call.
        try:
            response = self.client.messages.create(
                model=model_name,
                messages=anthropic_messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
                tools=tools_param,
                system=system_messages,
                # Pass system message as top-level parameter
                **kwargs
            )
        except Exception as e:
            raise APIError(f"Anthropic completion error: {e}")

        response_message = _process_response(response=response,
                                             toolset=toolset)

        messages.append(response_message)
        return response_message
