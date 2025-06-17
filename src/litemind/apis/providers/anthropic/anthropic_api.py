import os
from typing import List, Optional, Sequence, Type, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.builtin_tools.mcp_tool import BuiltinMCPTool
from litemind.agent.tools.builtin_tools.web_search_tool import BuiltinWebSearchTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.feature_scanner import get_default_model_feature_scanner
from litemind.apis.providers.anthropic.utils.check_availability import (
    check_anthropic_api_availability,
)
from litemind.apis.providers.anthropic.utils.convert_messages import (
    convert_messages_for_anthropic,
)
from litemind.apis.providers.anthropic.utils.format_tools import (
    format_tools_for_anthropic,
)
from litemind.apis.providers.anthropic.utils.list_models import (
    _get_anthropic_models_list,
)
from litemind.apis.providers.anthropic.utils.process_response import (
    process_response_from_anthropic,
)
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_text import Text

# ------------------------------- #
#   CONSTANTS (docs 2025-06-08)   #
# ------------------------------- #
_CTX_200K = 200_000
_CTX_100K = 100_000

_OUT_64K = 64_000
_OUT_32K = 32_000
_OUT_8K = 8_192
_OUT_4K = 4_096
_OUT_128K = 128_000  # beta header for Claude-3.7-Sonnet only


class AnthropicApi(DefaultApi):
    """
    An enhanced Anthropic API implementation that conforms to the `BaseApi` abstract interface.

    This implementation supports the latest Anthropic API features including:
    - Built-in web search tool
    - Model Context Protocol (MCP) connector
    - Code execution tool (sandbox Python execution)
    - Extended thinking capabilities
    - Interleaved thinking between tool calls

    Anthropic models support text generation, image inputs, pdf inputs, and thinking,
    but do not support audio, video, or embeddings natively. Support for these features
    are provided by the Litemind API via our fallback mechanism.

    Set the ANTHROPIC_API_KEY environment variable to your Anthropic API key.
    You can get it from https://console.anthropic.com/account/api-keys.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        allow_media_conversions: bool = True,
        allow_media_conversions_with_models: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        **anthropic_api_kwargs,
    ):
        """
        Initialize the Anthropic client.

        Parameters
        ----------
        api_key : Optional[str]
            The API key for Anthropic. If not provided, we'll read from ANTHROPIC_API_KEY env var.
        base_url : Optional[str]
            The base URL for the Anthropic API.
        allow_media_conversions: bool
            If True, the API will allow media conversions using the default media converter.
        allow_media_conversions_with_models: bool
            If True, the API will allow media conversions using models that support the required features.
        callback_manager: CallbackManager
            A callback manager to handle callbacks.
        **anthropic_api_kwargs
            Additional keyword arguments to pass to the Anthropic client.
        """
        super().__init__(
            allow_media_conversions=allow_media_conversions,
            allow_media_conversions_with_models=allow_media_conversions_with_models,
            callback_manager=callback_manager,
        )

        # Import Anthropic
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Please install anthropic to use AnthropicApi: pip install anthropic"
            )

        # Call superclass constructor:
        self.feature_scanner = get_default_model_feature_scanner()

        # Set API key:
        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")

        # If no API key is provided, raise an error:
        if not api_key:
            raise APIError(
                "An Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY or pass `api_key=...` explicitly."
            )

        # Save the API key:
        self.api_key = api_key

        # Save base URL:
        self.base_url = base_url

        # Save additional kwargs:
        self.anthropic_api_kwargs = anthropic_api_kwargs

        try:

            # Initialize the feature scanner:
            self.feature_scanner = get_default_model_feature_scanner()

            # Create the Anthropic client
            from anthropic import Anthropic

            self.client = Anthropic(
                api_key=self.api_key, base_url=self.base_url, **anthropic_api_kwargs
            )

            # Fetch the raw model list:
            self._model_list = _get_anthropic_models_list(self.client)

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing Anthropic client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:
        """Check if the API is available and credentials are valid."""
        # Check if the API key is provided:
        if api_key is not None:
            from anthropic import Anthropic

            client = Anthropic(
                api_key=api_key, base_url=self.base_url, **self.anthropic_api_kwargs
            )
        else:
            client = self.client

        # get model list without thinking models:
        model_list = list(self._model_list)
        model_list = [model for model in model_list if "thinking" not in model]

        # Get one model name:
        model_name = model_list[0]

        result = check_anthropic_api_availability(client, model_name)

        # Call the callback manager:
        self.callback_manager.on_availability_check(result)

        # Return the result:
        return result

    def list_models(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
    ) -> List[str]:

        try:
            # Normalize the features:
            features = ModelFeatures.normalise(features)
            non_features = ModelFeatures.normalise(non_features)

            # Get the model list:
            model_list = list(self._model_list)

            # Add the models from the super class:
            model_list += super().list_models()

            # Filter the models based on the features:
            if features:
                model_list = self._filter_models(
                    model_list,
                    features=features,
                    non_features=non_features,
                    media_types=media_types,
                )

            # Call _callbacks:
            self.callback_manager.on_model_list(model_list)

            return model_list

        except Exception:
            raise APIError("Error fetching model list from Anthropic.")

    def get_best_model(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        exclusion_filters: Optional[Union[str, List[str]]] = None,
    ) -> Optional[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)
        non_features = ModelFeatures.normalise(non_features)

        # Get the list of models:
        model_list = self.list_models()

        # Filter models based on requirements:
        model_list = self._filter_models(
            model_list,
            features=features,
            non_features=non_features,
            media_types=media_types,
            exclusion_filters=exclusion_filters,
        )

        # If we have any models left, return the first one
        if model_list:
            model_name = model_list[0]
        else:
            model_name = None

        # Call the callbacks:
        self.callback_manager.on_best_model_selected(model_name)

        return model_name

    def has_model_support_for(
        self,
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]],
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        model_name: Optional[str] = None,
    ) -> bool:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=features, media_types=media_types)

        # If model_name is None then we return False:
        if model_name is None:
            return False

        # We check if the superclass says that the model supports the features:
        if super().has_model_support_for(
            features=features, media_types=media_types, model_name=model_name
        ):
            return True

        # General case we pull info from the scanner:
        for feature in features:
            if not self.feature_scanner.supports_feature(
                self.__class__, model_name, feature
            ):
                # If the model does not support the feature, we return False:
                return False

        return True

    def _has_cache_support(self, model_name: str) -> bool:
        if "sonnet" in model_name or "claude-2.0" in model_name:
            return False
        return True

    def _has_web_search_support(self, model_name: str) -> bool:
        """Check if model supports built-in web search."""
        # Web search is available on most modern Claude models
        if any(
            x in model_name.lower()
            for x in ["claude-4", "claude-3", "claude-sonnet", "claude-opus"]
        ):
            return True
        return False

    def _has_mcp_support(self, model_name: str) -> bool:
        """Check if model supports MCP connector."""
        # MCP is available on Claude 4 and recent Claude 3 models
        if any(
            x in model_name.lower()
            for x in ["claude-4", "claude-sonnet-4", "claude-opus-4"]
        ):
            return True
        return False

    def _has_code_execution_support(self, model_name: str) -> bool:
        """Check if model supports code execution tool."""
        # Code execution is available on Claude 4 models
        if any(
            x in model_name.lower()
            for x in ["claude-4", "claude-sonnet-4", "claude-opus-4"]
        ):
            return True
        return False

    def _has_thinking_support(self, model_name: str) -> bool:
        """Check if model supports extended thinking."""
        # Thinking is available on Claude 4 and Claude 3.7+ models
        if any(
            x in model_name.lower()
            for x in ["claude-4", "claude-3-7", "claude-sonnet-3-7"]
        ):
            return True
        return False

    # ------------------------------- #
    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Maximum *total* context (input + output) tokens for an Anthropic Claude model.
        Falls back to 100 K for unknown models.
        """
        if model_name is None:
            model_name = self.get_best_model()
        name = model_name.lower()

        # ---- Claude 4 ----
        if "opus-4" in name or "sonnet-4" in name:
            return _CTX_200K

        # ---- Claude 3.7 / 3.5 / 3 ----
        if "claude-3" in name:
            return _CTX_200K

        # ---- Claude 2.1 ----
        if "claude-2.1" in name:
            return _CTX_200K

        # ---- Claude 2 / Instant 1.x ----
        if "claude-2.0" in name or "claude-2" in name or "instant-1" in name:
            return _CTX_100K

        # ---- Fallback ----
        return _CTX_100K

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Documented *maximum* tokens Claude may generate in one response.
        Returns the highest published cap (may require beta header, see notes).
        Falls back to 4 096 for unknown models.
        """
        if model_name is None:
            model_name = self.get_best_model()
        name = model_name.lower()

        # ---- Claude 4 ----
        if "opus-4" in name:
            return _OUT_32K
        if "sonnet-4" in name:
            return _OUT_64K

        # ---- Claude 3.7 Sonnet ----
        if "3-7-sonnet" in name:
            return _OUT_64K  # can be _OUT_128K  but requires `output-128k-2025-02-19` header

        # ---- Claude 3.5 family ----
        if "3-5-sonnet" in name or "3-5-haiku" in name:
            return _OUT_8K

        # ---- Claude 3 base ----
        if "3-opus" in name or "3-sonnet" in name or "3-haiku" in name:
            return _OUT_4K

        # ---- Claude 2.1 ----
        if "claude-2.1" in name:
            return _OUT_4K

        # ---- Claude 2 / Instant 1.x ----
        if "claude-2.0" in name or "claude-2" in name or "instant-1" in name:
            return _OUT_4K

        # ---- Fallback ----
        return _OUT_4K

    def _format_tools_and_mcp_for_anthropic(self, toolset: ToolSet) -> tuple:
        """
        Format tools and MCP servers for Anthropic API.

        Returns
        -------
        tuple
            (anthropic_tools, mcp_servers, beta_headers)
        """
        from anthropic import NotGiven

        anthropic_tools = []
        mcp_servers = []
        beta_headers = {}

        if toolset:
            # Handle custom function tools using existing formatter
            custom_tools = format_tools_for_anthropic(toolset)
            if custom_tools:
                anthropic_tools = custom_tools

            # Handle built-in tools
            for builtin_tool in toolset.list_builtin_tools():
                if isinstance(builtin_tool, BuiltinWebSearchTool):
                    # Add web search tool
                    web_search_tool: BuiltinWebSearchTool = builtin_tool
                    max_uses = (
                        web_search_tool.max_web_searches
                    )  # Default max uses for web search

                    anthropic_tools.append(
                        {
                            "type": "web_search_20250305",
                            "name": "web_search",
                            "max_uses": max_uses,
                        }
                    )

                elif isinstance(builtin_tool, BuiltinMCPTool):
                    # Add MCP server
                    mcp_tool: BuiltinMCPTool = builtin_tool
                    mcp_server = {
                        "type": "url",
                        "url": mcp_tool.server_url,
                        "name": mcp_tool.server_name,
                    }

                    # Add optional parameters
                    if mcp_tool.headers:
                        for key, value in mcp_tool.headers.items():
                            if key == "authorization_token":
                                # Special case for authorization token
                                mcp_server["authorization_token"] = value
                            else:
                                # Any other headers:
                                mcp_server[key] = value

                    if mcp_tool.allowed_tools:
                        mcp_server["tool_configuration"] = {
                            "enabled": True,
                            "allowed_tools": mcp_tool.allowed_tools,
                        }

                    mcp_servers.append(mcp_server)

                else:
                    # Throw an exception if unknown built-in tool is used:
                    raise ValueError(f"Unknown built-in tool: {builtin_tool.name}")

            # Set beta headers if needed
            if mcp_servers:
                beta_headers["anthropic-beta"] = "mcp-client-2025-04-04"

        if not anthropic_tools:
            # If no custom tools were provided, set to NotGiven
            anthropic_tools = NotGiven()

        return anthropic_tools, mcp_servers, beta_headers

    def generate_text(
        self,
        messages: List[Message],
        model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_num_output_tokens: Optional[int] = None,
        toolset: Optional[ToolSet] = None,
        use_tools: bool = True,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ) -> List[Message]:

        # Local variables for extended features with defaults
        use_code_execution = False  # Could be enabled via toolset or kwargs in future
        use_interleaved_thinking = True  # Could be enabled via kwargs in future

        # validate inputs using parent class method:
        super().generate_text(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_num_output_tokens=max_num_output_tokens,
            toolset=toolset,
            use_tools=use_tools,
            response_format=response_format,
            **kwargs,
        )

        # Anthropic imports:
        from anthropic import NotGiven

        # Set default model if not provided
        if model_name is None:
            model_name = self._get_best_model_for_text_generation(
                messages, toolset if use_tools else None, response_format
            )

        # Specific to Anthropic: extract system message if present
        system_messages = ""
        non_system_messages = []
        for message in messages:
            if message.role == "system":
                for block in message.blocks:
                    if block.has_type(Text):
                        text_media: Text = block.media
                        system_messages += text_media.text
                    else:
                        raise ValueError(
                            "System message should only contain text blocks."
                        )
            else:
                non_system_messages.append(message)

        # Preprocess the messages, we use non-system messages only:
        preprocessed_messages = self._preprocess_messages(
            messages=non_system_messages,
            allowed_media_types=self._get_allowed_media_types_for_text_generation(
                model_name=model_name
            ),
            exclude_extensions=["pdf"],
        )

        # Get max num of output tokens for model if not provided:
        if max_num_output_tokens is None:
            max_num_output_tokens = self.max_num_output_tokens(model_name)

        # Handle thinking mode (preserve original functionality)
        original_model_name = model_name
        if model_name.endswith("-thinking-high"):
            budget_tokens = max(1000, max_num_output_tokens // 2 - 1000)
            thinking = {"type": "enabled", "budget_tokens": budget_tokens}
            model_name = model_name.replace("-thinking-high", "")
            temperature = 1.0
            use_thinking = True
        elif model_name.endswith("-thinking-mid"):
            budget_tokens = max(1000, max_num_output_tokens // 2 - 1000)
            thinking = {"type": "enabled", "budget_tokens": budget_tokens}
            model_name = model_name.replace("-thinking-mid", "")
            temperature = 1.0
            use_thinking = True
        elif model_name.endswith("-thinking-low"):
            budget_tokens = max(1000, max_num_output_tokens // 4 - 1000)
            thinking = {"type": "enabled", "budget_tokens": budget_tokens}
            model_name = model_name.replace("-thinking-low", "")
            temperature = 1.0
            use_thinking = True
        else:
            thinking = NotGiven()

        # Format tools and MCP servers for Anthropic API
        anthropic_tools, mcp_servers, beta_headers = (
            self._format_tools_and_mcp_for_anthropic(toolset)
        )

        # Add code execution tool if enabled
        if use_code_execution:
            if anthropic_tools == NotGiven():
                anthropic_tools = []
            anthropic_tools.append(
                {"type": "code_execution_20241024", "name": "code_execution"}
            )

        # Set up interleaved thinking beta header if needed
        if use_interleaved_thinking:
            if beta_headers:
                beta_headers["anthropic-beta"] += ",interleaved-thinking-2025-05-14"
            else:
                beta_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        # List of new messages part of the response:
        new_messages = []

        try:
            # Loop until we get a response that doesn't require tool use :
            while True:

                # Convert remaining non-system litemind Messages to Anthropic messages:
                anthropic_formatted_messages = convert_messages_for_anthropic(
                    preprocessed_messages,
                    response_format=response_format,
                    cache_support=self._has_cache_support(model_name),
                    media_converter=self.media_converter,
                )

                # Prepare request parameters
                request_params = {
                    "model": model_name,
                    "messages": anthropic_formatted_messages,
                    "temperature": temperature,
                    "max_tokens": max_num_output_tokens,
                    "tools": anthropic_tools,
                    "system": system_messages,
                    "thinking": thinking,
                    "extra_headers": beta_headers if beta_headers else NotGiven(),
                    **kwargs,
                }

                # Add MCP servers if any
                if mcp_servers:
                    request_params["mcp_servers"] = mcp_servers

                # Pick the right function for streaming:
                anthropic_stream = (
                    self.client.beta.messages.stream
                    if mcp_servers
                    else self.client.messages.stream
                )

                # Call the Anthropic API:
                with anthropic_stream(
                    **request_params,
                ) as streaming_response:
                    # 1) Stream and handle partial events:
                    for event in streaming_response:
                        if event.type == "text":
                            # Call the callback manager for text streaming events:
                            self.callback_manager.on_text_streaming(event.text)

                    # 2) Get the final response:
                    anthropic_response = streaming_response.get_final_message()

                # Process the response from Anthropic:
                response = process_response_from_anthropic(
                    anthropic_response=anthropic_response,
                    response_format=response_format,
                )

                # Append response message to original, preprocessed, and new messages:
                messages.append(response)
                preprocessed_messages.append(response)
                new_messages.append(response)

                # If the model wants to use a tool, parse out the tool calls:
                if not response.has(Action):
                    # Break out of the loop if no tool use is required anymore:
                    break

                if not use_tools:
                    # Break out of the loop if we're not using tools:
                    break

                # Process the tool calls for custom function tools only
                # Built-in tools (web search, MCP) are handled automatically by Anthropic
                if toolset:
                    custom_tools_used = False
                    for block in response.blocks:
                        if block.has_type(Action):
                            action = block.media.action
                            if hasattr(action, "tool_name"):
                                # Check if this is a custom tool (not built-in)
                                tool: Optional[BaseTool] = toolset.get_tool(
                                    action.tool_name
                                )
                                if tool and not tool.is_builtin():
                                    custom_tools_used = True
                                    break

                    # Only process tool calls if custom tools were used
                    if custom_tools_used:
                        self._process_tool_calls(
                            response,
                            messages,
                            new_messages,
                            preprocessed_messages,
                            toolset,
                        )
                    else:
                        # Built-in tools are handled automatically, just break
                        break
                else:
                    break

            # Add all parameters to kwargs:
            kwargs.update(
                {
                    "model_name": original_model_name,  # Use original model name for callbacks
                    "temperature": temperature,
                    "max_output_tokens": max_num_output_tokens,
                    "toolset": toolset,
                    "response_format": response_format,
                }
            )

            # Call the callback manager:
            self.callback_manager.on_text_generation(
                messages=messages, response=response, **kwargs
            )

        except Exception as e:
            # print stacktrace:
            import traceback

            traceback.print_exc()
            # Raise an APIError with the exception message:
            raise APIError(f"Anthropic generate text error: {e}")

        return new_messages
