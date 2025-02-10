import os
from typing import List, Optional, Sequence, Union

from pydantic import BaseModel

from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.anthropic.utils.convert_messages import \
    convert_messages_for_anthropic
from litemind.apis.anthropic.utils.format_tools import format_tools_for_anthropic
from litemind.apis.anthropic.utils.process_response import process_response_from_anthropic
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError


class AnthropicApi(DefaultApi):
    """
    An Anthropic API implementation that conforms to the `BaseApi` abstract interface.
    Uses the `anthropic` library's `client.messages.create(...)` methods for completions.
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 callback_manager: Optional[CallbackManager] = None,
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

        super().__init__(callback_manager=callback_manager)

        if api_key is None:
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise APIError(
                "An Anthropic API key is required. "
                "Set ANTHROPIC_API_KEY or pass `api_key=...` explicitly."
            )

        try:
            # Create the Anthropic client
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=api_key,
                **kwargs  # e.g. timeout=30.0, max_retries=2, etc.
            )
        except Exception as e:
            # Print stack trace:
            import traceback
            traceback.print_exc()
            raise APINotAvailableError(
                f"Error initializing Anthropic client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[
        str] = None) -> bool:

        # Use the provided key if any, else the one from the client
        candidate_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not candidate_key:
            self.callback_manager.on_availability_check(False)
            return False

        # We'll attempt a trivial request
        try:
            test_message = {
                "role": "user",
                "content": "Hello, is this key valid?",
            }
            resp = self.client.messages.create(
                model=self.get_best_model(features=ModelFeatures.TextGeneration),
                max_tokens=16,
                messages=[test_message],
            )
            # If no exception: assume it's valid enough
            _ = resp.content  # Accessing content to ensure it exists
            self.callback_manager.on_availability_check(True)
            return True
        except Exception:
            # printout stacktrace:
            import traceback
            traceback.print_exc()
            self.callback_manager.on_availability_check(False)
            return False

    def list_models(self, features: Optional[Sequence[ModelFeatures]] = None) -> \
            List[str]:

        try:
            # List Anthropic models:
            from anthropic.types import ModelInfo
            from openai.pagination import SyncPage

            # Get the first 100 models, to be safe:
            models_info: List[ModelInfo] = self.client.models.list(
                limit=100).data

            # Extract model IDs
            model_list: List[str] = list([str(info.id) for info in models_info])

            # Add the models from the super class:
            model_list += super().list_models()

            # Filter the models based on the features:
            if features:
                model_list = self._filter_models(model_list, features=features)

            # Call callbacks:
            self.callback_manager.on_model_list(model_list)

            return model_list

        except Exception:
            raise APIError("Error fetching model list from Anthropic.")

    def get_best_model(self,
                       features: Optional[Union[
                           str, List[str], ModelFeatures, Sequence[ModelFeatures]]] = None,
                       exclusion_filters: Optional[Union[str, List[str]]] = None) -> \
            Optional[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the list of models:
        model_list = self.list_models()

        # Filter models based on requirements:
        # Filter the models based on the requirements:
        model_list = self._filter_models(model_list,
                                         features=features,
                                         exclusion_filters=exclusion_filters)

        # If we have any models left, return the first one
        if model_list:
            model_name = model_list[0]
        else:
            model_name = None

        # Call the callbacks:
        self.callback_manager.on_best_model_selected(model_name)

        return model_name

    def has_model_support_for(self,
                              features: Union[
                                  str, List[str], ModelFeatures, Sequence[
                                      ModelFeatures]],
                              model_name: Optional[str] = None) -> bool:

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
        if not 'claude' in model_name.lower():
            return False
        return True

    def _has_image_support(self, model_name: str) -> bool:
        if 'claude-2.0' in model_name or 'haiku' in model_name:
            return False
        if "3-5" in model_name or "sonnet" in model_name or "vision" in model_name:
            return True
        return False

    def _has_audio_support(self, model_name: Optional[str] = None) -> bool:

        # No Anthropic models currently support Audio, but we use local whisper as a fallback:
        return False

    def _has_document_support(self, model_name: str) -> bool:
        return (self.has_model_support_for([ModelFeatures.Image,
                                            ModelFeatures.TextGeneration],
                                           model_name)
                and self.has_model_support_for(ModelFeatures.VideoConversion))

    def _has_tool_support(self, model_name: str) -> bool:

        return "claude-3-5" in model_name

    def _has_structured_output_support(self, model_name: Optional[str] = None) -> bool:

        return "claude-3" in model_name

    def _has_cache_support(self, model_name: str) -> bool:
        if 'sonnet' in model_name or 'claude-2.0' in model_name:
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
            model_name = self.get_best_model()

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

    def generate_text(self,
                      messages: List[Message],
                      model_name: Optional[str] = None,
                      temperature: float = 0.0,
                      max_output_tokens: Optional[int] = None,
                      toolset: Optional[ToolSet] = None,
                      response_format: Optional[BaseModel] = None,
                      **kwargs) -> Message:

        from anthropic import NotGiven

        # Extract system message if present
        system_messages = ''
        non_system_messages = []
        for message in messages:
            if message.role == "system":
                for block in message.blocks:
                    if block.block_type == BlockType.Text:
                        system_messages += block.content
                    else:
                        raise ValueError(
                            "System message should only contain text blocks.")
            else:
                non_system_messages.append(message)

        # Preprocess the messages, we use non-system messages only:
        preprocessed_messages = self._preprocess_messages(messages=non_system_messages)

        # Convert remaining non-system litemind Messages to Anthropic messages:
        anthropic_messages = convert_messages_for_anthropic(
            preprocessed_messages,
            response_format=response_format,
            cache_support=self._has_cache_support(model_name))

        # Convert a ToolSet to Anthropic "tools" param if any
        tools_param = format_tools_for_anthropic(
            toolset) if toolset else NotGiven()

        # Get max num of output tokens for model if not provided:
        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        try:
            # Call the Anthropic API:
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

        # Process the response:
        response_message = process_response_from_anthropic(response=response,
                                                           toolset=toolset,
                                                           response_format=response_format)

        # Append response message to messages:
        messages.append(response_message)

        # Add all parameters to kwargs:
        kwargs.update({
            'model_name': model_name,
            'temperature': temperature,
            'max_output_tokens': max_output_tokens,
            'toolset': toolset,
            'response_format': response_format
        })

        # Call the callback manager:
        self.callback_manager.on_text_generation(messages=messages,
                                                 response=response,
                                                 **kwargs)

        return response_message
