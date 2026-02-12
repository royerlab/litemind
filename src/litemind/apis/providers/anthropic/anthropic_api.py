"""Anthropic API implementation for the litemind unified API layer."""

import os
from typing import List, Optional, Sequence, Type, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.base_tool import BaseTool
from litemind.agent.tools.builtin_tools.mcp_tool import BuiltinMCPTool
from litemind.agent.tools.builtin_tools.web_search_tool import BuiltinWebSearchTool
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.api_callback_manager import ApiCallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.model_registry import get_default_model_registry
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
# Fallback values when model is not in registry
_DEFAULT_CONTEXT_WINDOW = 200_000
_DEFAULT_MAX_OUTPUT_TOKENS = 8_192

# Per Anthropic API docs: minimum thinking budget is 1024 tokens
_THINKING_BUDGET_MIN = 1024


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
        callback_manager: Optional[ApiCallbackManager] = None,
        enable_long_context: bool = True,
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
        callback_manager: ApiCallbackManager
            A callback manager to handle callbacks.
        enable_long_context : bool
            If True, includes '-1m' model variants in model listings for supported
            models (Opus 4.6, Sonnet 4.5, Sonnet 4). Selecting a '-1m' variant
            automatically sends the context-1m beta header for 1M token context.
            Long context pricing applies to requests exceeding 200K tokens.
            Default is True.
        **anthropic_api_kwargs
            Additional keyword arguments to pass to the Anthropic client.
        """
        super().__init__(
            allow_media_conversions=allow_media_conversions,
            allow_media_conversions_with_models=allow_media_conversions_with_models,
            callback_manager=callback_manager,
        )

        # Check if anthropic package is installed
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise ImportError(
                "Please install anthropic to use AnthropicApi: pip install anthropic"
            )

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

        # Save long context setting:
        self.enable_long_context = enable_long_context

        # Save additional kwargs:
        self.anthropic_api_kwargs = anthropic_api_kwargs

        try:
            # Initialize the model registry for feature lookups:
            self.model_registry = get_default_model_registry()

            # Create the Anthropic client
            from anthropic import Anthropic

            # ensure the timeout parameter is set in kwargs and high enough:
            if (
                "timeout" not in self.anthropic_api_kwargs
                or self.anthropic_api_kwargs["timeout"] < 6000
            ):
                self.anthropic_api_kwargs["timeout"] = 6000

            self.client = Anthropic(
                api_key=self.api_key, base_url=self.base_url, **anthropic_api_kwargs
            )

            # Fetch the raw model list:
            self._model_list = _get_anthropic_models_list(
                self.client, enable_long_context=self.enable_long_context
            )

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing Anthropic client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:
        """Check if the Anthropic API is available and credentials are valid.

        Parameters
        ----------
        api_key : Optional[str]
            If provided, creates a new client with this key for the check.
            Otherwise uses the existing client.

        Returns
        -------
        bool
            True if the API is available and responding, False otherwise.
        """
        # Check if the API key is provided:
        if api_key is not None:
            from anthropic import Anthropic

            client = Anthropic(
                api_key=api_key, base_url=self.base_url, **self.anthropic_api_kwargs
            )
        else:
            client = self.client

        # get model list without thinking or -1m variant models:
        model_list = list(self._model_list)
        model_list = [
            model
            for model in model_list
            if "thinking" not in model and "-1m" not in model
        ]

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
        """List available Anthropic models, optionally filtered by features.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required features to filter models by.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features that models must NOT have.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Required media type support to filter by.

        Returns
        -------
        List[str]
            List of model name strings matching the criteria.

        Raises
        ------
        APIError
            If fetching the model list fails.
        """

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
        """Select the best available Anthropic model matching the given criteria.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required features the model must support.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features the model must NOT support.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Required media type support.
        exclusion_filters : Optional[Union[str, List[str]]]
            Substring patterns to exclude from model names.

        Returns
        -------
        Optional[str]
            The name of the best matching model, or None if no match found.
        """

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
        """Check whether a model supports all of the requested features.

        Parameters
        ----------
        features : Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            Features to check for.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must support.
        model_name : Optional[str]
            Specific model to check. If None, uses ``get_best_model``.

        Returns
        -------
        bool
            True if the model supports all requested features.
        """

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

        # For registry lookup, try exact name first. If model not in registry,
        # strip thinking suffix (handles combined -1m-thinking-* models
        # like claude-opus-4-6-1m-thinking-high → claude-opus-4-6-1m)
        has_thinking_suffix = any(
            model_name.endswith(s)
            for s in [
                "-thinking-max",
                "-thinking-high",
                "-thinking-mid",
                "-thinking-low",
            ]
        )
        registry_name = model_name
        if not self.model_registry.get_model_info(self.__class__, registry_name):
            registry_name = self._strip_thinking_suffix(model_name)

        # Check feature support from the registry:
        for feature in features:
            # Models with a thinking suffix always support Thinking
            if feature == ModelFeatures.Thinking and has_thinking_suffix:
                continue
            if not self.model_registry.supports_feature(
                self.__class__, registry_name, feature
            ):
                # If the model does not support the feature, we return False:
                return False

        return True

    def _strip_thinking_suffix(self, model_name: str) -> str:
        """Strip thinking suffix from model name for registry lookup.

        Parameters
        ----------
        model_name : str
            Model name that may contain a thinking suffix.

        Returns
        -------
        str
            Model name with the thinking suffix removed, or the original
            name if no suffix was found.
        """
        for suffix in [
            "-thinking-max",
            "-thinking-high",
            "-thinking-mid",
            "-thinking-low",
        ]:
            if model_name.endswith(suffix):
                return model_name[: -len(suffix)]
        return model_name

    def _has_cache_support(self, model_name: str) -> bool:
        """Check if a model supports prompt caching.

        Supported: Claude 3.5+, 3.7+, 4+, 4.5+ (all variants).
        Not supported: Claude 2.x, Claude 3 base (3-opus, 3-sonnet, 3-haiku
        without .5 or .7).

        Parameters
        ----------
        model_name : str
            The model name to check.

        Returns
        -------
        bool
            True if the model supports prompt caching.
        """
        name = model_name.lower()

        # Claude 2.x - no caching
        if "claude-2" in name:
            return False

        # Claude 3 base models (not 3.5 or 3.7) don't support caching
        # Check for exact claude-3-opus, claude-3-sonnet, claude-3-haiku patterns
        # but exclude claude-3-5 and claude-3-7 which DO support caching
        if any(
            x in name for x in ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"]
        ):
            # These are Claude 3 base models, but 3.5 and 3.7 support caching
            if "claude-3-5" not in name and "claude-3-7" not in name:
                return False

        # All modern models (3.5+, 3.7, 4, 4.5) support caching
        return True

    def _uses_adaptive_thinking(self, model_name: str) -> bool:
        """Check if a model uses adaptive thinking instead of budget_tokens.

        Currently only Claude Opus 4.6 supports adaptive thinking.
        Older models require ``type: "enabled"`` with ``budget_tokens``.

        Parameters
        ----------
        model_name : str
            The model name to check (thinking suffix should already be stripped).

        Returns
        -------
        bool
            True if the model supports adaptive thinking.
        """
        base = self._strip_thinking_suffix(model_name).lower()
        return "claude-opus-4-6" in base

    # ------------------------------- #
    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """Return the maximum context window (input tokens) for a Claude model.

        Uses the model registry for accurate, curated values. Falls back to
        200K for unknown models. Models with the ``-1m`` suffix return 1M
        tokens (from registry).

        Parameters
        ----------
        model_name : Optional[str]
            The model to query. If None, uses ``get_best_model``.

        Returns
        -------
        int
            Maximum number of input tokens the model accepts.
        """
        if model_name is None:
            model_name = self.get_best_model()

        # Strip thinking suffix for registry lookup (keep -1m suffix)
        base_model = self._strip_thinking_suffix(model_name)

        # Try registry first (most accurate source)
        model_info = self.model_registry.get_model_info(self.__class__, base_model)
        if model_info and model_info.context_window:
            return model_info.context_window

        # Fallback for unknown models
        return _DEFAULT_CONTEXT_WINDOW

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """Return the maximum output tokens a Claude model may generate.

        Uses the model registry for accurate, curated values. Falls back to
        8K for unknown models.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query. If None, uses ``get_best_model``.

        Returns
        -------
        int
            Maximum number of output tokens the model can produce.
        """
        if model_name is None:
            model_name = self.get_best_model()

        # Strip thinking suffix for registry lookup
        base_model = self._strip_thinking_suffix(model_name)

        # Try registry first (most accurate source)
        model_info = self.model_registry.get_model_info(self.__class__, base_model)
        if model_info and model_info.max_output_tokens:
            return model_info.max_output_tokens

        # Fallback for unknown models
        return _DEFAULT_MAX_OUTPUT_TOKENS

    def _format_tools_and_mcp_for_anthropic(self, toolset: ToolSet) -> tuple:
        """Format tools and MCP servers for the Anthropic API.

        Separates custom function tools, built-in web search tools, and MCP
        server configurations into their respective Anthropic API formats.

        Parameters
        ----------
        toolset : ToolSet
            The toolset containing function tools, built-in tools, and MCP tools.

        Returns
        -------
        tuple
            A 3-tuple of ``(anthropic_tools, mcp_servers, beta_headers)`` where
            ``anthropic_tools`` is a list of tool definitions (or ``NotGiven``),
            ``mcp_servers`` is a list of MCP server configs, and
            ``beta_headers`` is a dict of beta feature headers to send.

        Raises
        ------
        ValueError
            If an unsupported built-in tool type is encountered.
        """
        from anthropic import NotGiven

        anthropic_tools = []
        mcp_servers = []
        beta_features = []  # List of beta feature strings to join later

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

                    # Build web search config with required parameters
                    web_search_config = {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "max_uses": web_search_tool.max_web_searches,
                    }

                    # Add optional allowed_domains if specified
                    if web_search_tool.allowed_domains:
                        web_search_config["allowed_domains"] = (
                            web_search_tool.allowed_domains
                        )

                    anthropic_tools.append(web_search_config)

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

            # Add MCP beta feature if needed
            if mcp_servers:
                beta_features.append("mcp-client-2025-04-04")

        if not anthropic_tools:
            # If no custom tools were provided, set to NotGiven
            anthropic_tools = NotGiven()

        # Build beta headers from features list
        beta_headers = (
            {"anthropic-beta": ",".join(beta_features)} if beta_features else {}
        )

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
        tool_choice: Optional[Union[str, dict]] = None,
        **kwargs,
    ) -> List[Message]:
        """Generate text using the Anthropic API with tool and thinking support.

        Converts litemind messages to Anthropic format, handles extended thinking
        mode, MCP and web search built-in tools, streaming, and iterative
        tool-use loops.

        Parameters
        ----------
        messages : List[Message]
            Conversation messages including system, user, and assistant messages.
        model_name : Optional[str]
            Model to use. If None, auto-selects based on message content.
        temperature : float
            Sampling temperature (0.0 to 1.0). Overridden to 1.0 for thinking mode.
        max_num_output_tokens : Optional[int]
            Maximum tokens in the response. If None, uses the model default.
        toolset : Optional[ToolSet]
            Tools available for the model to call.
        use_tools : bool
            Whether to execute tool calls returned by the model.
        response_format : Optional[BaseModel]
            Pydantic model for structured JSON output.
        tool_choice : Optional[Union[str, dict]]
            Tool selection strategy: "auto", "any", "none", a tool name, or
            an Anthropic tool_choice dict.
        **kwargs
            Additional keyword arguments passed to the Anthropic API.

        Returns
        -------
        List[Message]
            New messages generated during this call (responses and tool results).

        Raises
        ------
        APIError
            If the Anthropic API call fails.
        """

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

        # Handle thinking mode - strip suffix and configure thinking
        # Opus 4.6 uses adaptive thinking + effort parameter (budget_tokens deprecated)
        # Older models use enabled thinking + budget_tokens
        original_model_name = model_name
        thinking = NotGiven()
        output_config = NotGiven()
        thinking_level = None

        # Detect thinking level from suffix:
        for suffix, level in [
            ("-thinking-max", "max"),
            ("-thinking-high", "high"),
            ("-thinking-mid", "mid"),
            ("-thinking-low", "low"),
        ]:
            if model_name.endswith(suffix):
                thinking_level = level
                model_name = model_name[: -len(suffix)]
                break

        if thinking_level:
            temperature = 1.0
            if self._uses_adaptive_thinking(model_name):
                # Opus 4.6+: adaptive thinking with effort parameter
                thinking = {"type": "adaptive"}
                effort_map = {
                    "max": "max",
                    "high": "high",
                    "mid": "medium",
                    "low": "low",
                }
                output_config = {"effort": effort_map[thinking_level]}
            else:
                # Older models: budget_tokens approach
                budget_divisors = {"high": 2, "mid": 3, "low": 4}
                if thinking_level == "max":
                    # max effort not supported on older models, fall back to high
                    thinking_level = "high"
                available_output = self.max_num_output_tokens(model_name)
                budget_tokens = max(
                    _THINKING_BUDGET_MIN,
                    available_output // budget_divisors[thinking_level],
                )
                thinking = {"type": "enabled", "budget_tokens": budget_tokens}

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

        # Add interleaved thinking beta feature if needed
        if use_interleaved_thinking:
            existing_features = (
                beta_headers.get("anthropic-beta", "").split(",")
                if beta_headers
                else []
            )
            existing_features = [f for f in existing_features if f]  # Remove empty
            existing_features.append("interleaved-thinking-2025-05-14")
            beta_headers["anthropic-beta"] = ",".join(existing_features)

        # Handle long context -1m suffix: strip suffix and add beta header
        if model_name.endswith("-1m"):
            model_name = model_name[:-3]  # Strip -1m suffix
            existing_features = (
                beta_headers.get("anthropic-beta", "").split(",")
                if beta_headers
                else []
            )
            existing_features = [f for f in existing_features if f]  # Remove empty
            existing_features.append("context-1m-2025-08-07")
            beta_headers["anthropic-beta"] = ",".join(existing_features)

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

                # Add output_config for adaptive thinking (Opus 4.6+).
                # Uses extra_body for SDK compatibility (output_config is
                # not a typed parameter in all SDK versions):
                if output_config is not NotGiven():
                    request_params["extra_body"] = {"output_config": output_config}

                # Add tool_choice if specified and tools are present
                if tool_choice is not None and anthropic_tools != NotGiven():
                    if isinstance(tool_choice, str):
                        if tool_choice in ["auto", "any"]:
                            request_params["tool_choice"] = {"type": tool_choice}
                        elif tool_choice == "none":
                            # Don't use tools even if provided
                            request_params["tool_choice"] = {"type": "none"}
                        else:
                            # Assume it's a specific tool name
                            request_params["tool_choice"] = {
                                "type": "tool",
                                "name": tool_choice,
                            }
                    elif isinstance(tool_choice, dict):
                        request_params["tool_choice"] = tool_choice

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
