import os
from typing import List, Optional, Sequence, Union

from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import (
    APIError,
    APINotAvailableError,
    FeatureNotAvailableError,
)
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
from litemind.media.types.media_action import Action
from litemind.media.types.media_text import Text


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
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ):
        """
        Initialize the Anthropic client.

        Parameters
        ----------
        api_key : Optional[str]
            The API key for Anthropic. If not provided, we'll read from ANTHROPIC_API_KEY env var.
        kwargs : dict
            Additional options (e.g. `timeout=...`, `max_retries=...`) passed to `Anthropic(...)`.
        """

        super().__init__(callback_manager=callback_manager)

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
        self.kwargs = kwargs

        try:
            # Create the Anthropic client
            from anthropic import Anthropic

            self.client = Anthropic(
                api_key=self.api_key, base_url=self.base_url, **kwargs
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

            client = Anthropic(api_key=api_key, base_url=self.base_url, **self.kwargs)
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
                    model_list, features=features, non_features=non_features
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
        model_name: Optional[str] = None,
    ) -> bool:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=features)

        # If model_name is None then we return False:
        if model_name is None:
            return False

        # We check if the superclass says that the model supports the features:
        if super().has_model_support_for(features=features, model_name=model_name):
            return True

        # Check that the model has all the required features:
        for feature in features:

            if feature == ModelFeatures.TextGeneration:
                if not self._has_text_gen_support(model_name):
                    return False

            elif feature == ModelFeatures.ImageGeneration:
                return False

            elif feature == ModelFeatures.Thinking:
                if not self._has_thinking_support(model_name):
                    return False

            elif feature == ModelFeatures.Image:
                if not self._has_image_support(model_name):
                    return False

            elif feature == ModelFeatures.Audio:
                if not self._has_audio_support(model_name):
                    return False

            elif feature == ModelFeatures.Video:
                if not self._has_image_support(model_name):
                    return False

            elif feature == ModelFeatures.Document:
                if not self._has_document_support(model_name):
                    return False

            elif feature == ModelFeatures.Tools:
                if not self._has_tool_support(model_name):
                    return False

            elif feature == ModelFeatures.StructuredTextGeneration:
                if not self._has_structured_output_support(model_name):
                    return False

            else:
                if not super().has_model_support_for(feature, model_name):
                    return False

        return True

    def _has_text_gen_support(self, model_name: str) -> bool:
        if not "claude" in model_name.lower():
            return False
        return True

    def _has_thinking_support(self, model_name: str) -> bool:
        if not "thinking" in model_name.lower():
            return False
        return True

    def _has_image_support(self, model_name: str) -> bool:
        if "claude-2.0" in model_name or "haiku" in model_name:
            return False
        if (
            "3-5" in model_name
            or "3-7" in model_name
            or "sonnet" in model_name
            or "vision" in model_name
        ):
            return True
        return False

    def _has_audio_support(self, model_name: Optional[str] = None) -> bool:

        # No Anthropic models currently support Audio, but we use local whisper as a fallback:
        return False

    def _has_document_support(self, model_name: str) -> bool:
        return self.has_model_support_for(
            [ModelFeatures.Image, ModelFeatures.TextGeneration], model_name
        ) and self.has_model_support_for(ModelFeatures.VideoConversion)

    def _has_tool_support(self, model_name: str) -> bool:

        return "claude-3-5" in model_name or "claude-3-7" in model_name

    def _has_structured_output_support(self, model_name: Optional[str] = None) -> bool:

        return "claude-3" in model_name

    def _has_cache_support(self, model_name: str) -> bool:
        if "sonnet" in model_name or "claude-2.0" in model_name:
            return False
        return True

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Return the maximum *input tokens* (context window) for an Anthropic Claude model.

        If model_name is unrecognized, fallback = 100_000 (safe guess).
        """

        if model_name is None:
            model_name = self.get_best_model()

        name = model_name.lower()

        # Claude 3.7 Sonnet => 200k
        if "claude-3-7-sonnet" in name:
            return 200_000

        # Claude 3.5 Sonnet/Haiku => 200k
        if "claude-3-5-sonnet" in name or "claude-3-5-haiku" in name:
            return 200_000

        # Claude 3 (Opus, Sonnet, Haiku) => 200k
        if (
            "claude-3-opus" in name
            or "claude-3-sonnet" in name
            or "claude-3-haiku" in name
        ):
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

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Return the maximum *output tokens* that can be generated by a given Anthropic Claude model.

        If model_name is unrecognized, fallback = 4096.
        """
        if model_name is None:
            model_name = self.get_best_model()

        name = model_name.lower()

        # Claude 3.7 Sonnet => 8192
        if "claude-3-7-sonnet" in name:
            return 8192

        # Claude 3.5 Sonnet/Haiku => 8192
        if "claude-3-5-sonnet" in name or "claude-3-5-haiku" in name:
            return 8192

        # Claude 3 Opus/Sonnet/Haiku => 4096
        if (
            "claude-3-opus" in name
            or "claude-3-sonnet" in name
            or "claude-3-haiku" in name
        ):
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

        # Anthropic imports:
        from anthropic import NotGiven

        # Set default model if not provided
        if model_name is None:
            # We Require the minimum features for text generation:
            features = [ModelFeatures.TextGeneration]
            # If tools are provided, we also require tools:
            if toolset:
                features.append(ModelFeatures.Tools)
            # If the messages contain media, we require the appropriate features:
            # TODO: implement
            model_name = self.get_best_model(features=features)

            if model_name is None:
                raise FeatureNotAvailableError(
                    f"No suitable model with features: {features}"
                )

        # Specific to Anthropic: extract system message if present
        system_messages = ""
        non_system_messages = []
        for message in messages:
            if message.role == "system":
                for block in message.blocks:
                    if block.has_type(Text):
                        system_messages += block.media.text
                    else:
                        raise ValueError(
                            "System message should only contain text blocks."
                        )
            else:
                non_system_messages.append(message)

        # Preprocess the messages, we use non-system messages only:
        preprocessed_messages = self._preprocess_messages(
            messages=non_system_messages, exclude_extensions=["pdf"]
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
            raise APIError(f"Anthropic generate text error: {e}")

        return new_messages
