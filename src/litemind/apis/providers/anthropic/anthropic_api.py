import os
from typing import List, Optional, Sequence, Type, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
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
    An Anthropic API implementation that conforms to the `BaseApi` abstract interface.

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
        allow_media_conversions: bool
            If True, the API will allow media conversions using the default media converter.
        allow_media_conversions_with_models: bool
            If True, the API will allow media conversions using models that support the required features in addition to the default media converter.
            To use this the allow_media_conversions parameter must be True.
        anthropic_api_kwargs : dict
            Additional options (e.g. `timeout=...`, `max_retries=...`) passed to `Anthropic(...)`.
        """

        super().__init__(
            allow_media_conversions=allow_media_conversions,
            allow_media_conversions_with_models=allow_media_conversions_with_models,
            callback_manager=callback_manager,
        )

        # Get the API key from the environment if not provided:
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
        # Filter the models based on the requirements:
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

        # Call the _callbacks:
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

        # validate inputs:
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

        # Convert a ToolSet to Anthropic "tools" attribute_key if any
        anthropic_tools = format_tools_for_anthropic(toolset) if toolset else NotGiven()

        # Thinking mode:
        if model_name.endswith("-thinking-high"):
            budget_tokens = max(1000, max_num_output_tokens // 2 - 1000)
            thinking = {"type": "enabled", "budget_tokens": budget_tokens}
            model_name = model_name.replace("-thinking-high", "")
            temperature = 1.0
        elif model_name.endswith("-thinking-mid"):
            budget_tokens = max(1000, max_num_output_tokens // 2 - 1000)
            thinking = {"type": "enabled", "budget_tokens": budget_tokens}
            model_name = model_name.replace("-thinking-mid", "")
            temperature = 1.0
        elif model_name.endswith("-thinking-low"):
            budget_tokens = max(1000, max_num_output_tokens // 4 - 1000)
            thinking = {"type": "enabled", "budget_tokens": budget_tokens}
            model_name = model_name.replace("-thinking-low", "")
            temperature = 1.0
        else:
            thinking = NotGiven()

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

                # Call the Anthropic API:
                with self.client.messages.stream(
                    model=model_name,
                    messages=anthropic_formatted_messages,
                    temperature=temperature,
                    max_tokens=max_num_output_tokens,
                    tools=anthropic_tools,
                    system=system_messages,
                    thinking=thinking,
                    # Pass system message as top-level parameter
                    **kwargs,
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

                # Process the tool calls:
                self._process_tool_calls(
                    response, messages, new_messages, preprocessed_messages, toolset
                )

            # Add all parameters to kwargs:
            kwargs.update(
                {
                    "model_name": model_name,
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
