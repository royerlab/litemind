import os
import tempfile
from io import BytesIO
from typing import List, Optional, Sequence, Union

import requests
from openai import OpenAI
from PIL import Image
from PIL.Image import Resampling
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.providers.openai.utils.check_availability import (
    check_openai_api_availability,
)
from litemind.apis.providers.openai.utils.convert_messages import (
    convert_messages_for_openai,
)
from litemind.apis.providers.openai.utils.format_tools import format_tools_for_openai
from litemind.apis.providers.openai.utils.list_models import (
    _get_raw_openai_model_list,
    get_openai_model_list,
)
from litemind.apis.providers.openai.utils.num_tokens import estimate_num_input_tokens
from litemind.apis.providers.openai.utils.preprocess_messages import (
    openai_preprocess_messages,
)
from litemind.apis.providers.openai.utils.process_response import (
    process_response_from_openai,
)
from litemind.media.types.media_action import Action
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class OpenAIApi(DefaultApi):
    """
    An OpenAI API implementation conforming to the BaseApi interface.

    OpenAI models support text generation, some support image and audio inputs and thinking,
    but do not support all features natively. Support for these features
    are provided by the Litemind API via our fallback mechanism.

    Set the OPENAI_API_KEY environment variable to your OpenAI API key, or pass it as an argument to the constructor.
    You can get your API key from https://platform.openai.com/account/api-keys.

    The OpenAIApi class provides a high-level interface for interacting with the OpenAI API.
    It supports various features such as text generation, image generation, audio processing, and more.
    The class also provides methods for checking the availability of the API, listing available models, and selecting the best model based on the required features.
    The OpenAIApi class is designed to be used with the OpenAI API, but it can also be used with other APIs that are compatible with the OpenAI API.
    The OpenAIApi class is a subclass of the DefaultApi class, which provides a default implementation of the BaseApi interface.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAI API client.
        Parameters
        ----------
        api_key: Optional[str]
            The API key for OpenAI. If not provided, we'll read from OPENAI_API_KEY env var.
        base_url: Optional[str]
            The base URL for the OpenAI or compatible API.
        use_whisper_for_audio_if_needed: bool
            If True, use whisper to transcribe audio if the model does not support audio.
        kwargs: dict
            Additional options (e.g. `timeout=...`, `max_retries=...`) passed to `OpenAI(...)`.
        """

        super().__init__(callback_manager=callback_manager)

        # get key from environmental variables:
        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")

        # raise error if api_key is None:
        if api_key is None:
            raise APIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the OPENAI_API_KEY environment variable"
            )

        # Save the API key:
        self.api_key = api_key

        # Save base URL:
        self.base_url = base_url

        # Save additional kwargs:
        self.kwargs = kwargs

        try:
            # Create an OpenAI client:
            from openai import OpenAI

            self.client: OpenAI = OpenAI(
                api_key=self.api_key, base_url=self.base_url, **self.kwargs
            )

            # Get the raw list of models:
            self._raw_model_list = _get_raw_openai_model_list(self.client)

            # Get the list of models from OpenAI:
            self._model_list = get_openai_model_list(self._raw_model_list)

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing OpenAI client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:

        # Check if the API key is provided:
        if api_key is not None:
            client = OpenAI(api_key=api_key, base_url=self.base_url, **self.kwargs)
        else:
            client = self.client

        # Check the OpenAI API key:
        result = check_openai_api_availability(client)

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
            # Normalise the features:
            features = ModelFeatures.normalise(features)
            non_features = ModelFeatures.normalise(non_features)

            # Get the models from OpenAI:
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
            raise APIError("Error fetching model list from OpenAI.")

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

        # Get model list:
        model_list = self.list_models()

        # Filter the models based on the requirements:
        model_list = self._filter_models(
            model_list,
            features=features,
            non_features=non_features,
            exclusion_filters=exclusion_filters,
        )

        # By default, result is None:
        best_model = None

        if len(model_list) == 0:
            pass
        else:
            best_model = model_list[0]

        # Call the callback manager:
        self.callback_manager.on_best_model_selected(best_model)

        return best_model

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

            elif feature == ModelFeatures.Thinking:
                if not self._has_thinking_support(model_name):
                    return False

            elif feature == ModelFeatures.AudioGeneration:
                pass

            elif feature == ModelFeatures.ImageGeneration:
                if not self._has_image_gen_support(model_name):
                    return False

            elif feature == ModelFeatures.TextEmbeddings:
                if not self._has_text_embed_support(model_name):
                    return False

            elif feature == ModelFeatures.ImageEmbeddings:

                if not self._has_text_embed_support(
                    model_name
                ) or not self.has_model_support_for(ModelFeatures.Image):
                    return False

            elif feature == ModelFeatures.AudioEmbeddings:

                if not self._has_text_embed_support(
                    model_name
                ) or not self.has_model_support_for(ModelFeatures.Audio):
                    return False

            elif feature == ModelFeatures.VideoEmbeddings:
                if not self._has_text_embed_support(
                    model_name
                ) or not self.has_model_support_for(ModelFeatures.Video):
                    return False

            elif feature == ModelFeatures.Image:
                if not self._has_image_support(model_name):
                    return False

            elif feature == ModelFeatures.Audio:
                if not self._has_audio_support(model_name):
                    return False

            elif feature == ModelFeatures.Video:
                if not self._has_video_support(model_name):
                    return False

            elif feature == ModelFeatures.Document:
                if not self._has_document_support(model_name):
                    return False

            elif feature == ModelFeatures.StructuredTextGeneration:
                if not self._has_structured_output_support(model_name):
                    return False

            elif feature == ModelFeatures.Tools:
                if not self._has_tools_support(model_name):
                    return False

            else:
                if not super().has_model_support_for(feature, model_name):
                    return False

        return True

    def _has_text_gen_support(self, model_name: str) -> bool:
        if (
            "text-embedding" in model_name
            or "dall-e" in model_name
            or "realtime" in model_name
            or "instruct" in model_name
            or "audio" in model_name
            or "transcribe" in model_name
        ):
            return False

        if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
            return True

        return False

    def _has_image_gen_support(self, model_name: str) -> bool:
        if "dall-e" not in model_name or "transcribe" in model_name:
            return False
        return True

    def _has_thinking_support(self, model_name: str) -> bool:

        if "o1" in model_name or "o3" in model_name:
            return True

        return False

    def _has_text_embed_support(self, model_name: str) -> bool:
        if "text-embedding" not in model_name:
            return False
        return True

    def _has_image_support(self, model_name: str) -> bool:

        # First, we check if it is a text-embedding model or image generation model:
        if (
            "text-embedding" in model_name
            or "dall-e" in model_name
            or "audio" in model_name
            or "gpt-3.5" in model_name
            or "realtime" in model_name
            or "o3-mini" in model_name
            or "o3-preview" in model_name
        ):
            return False

        # Then, we check if it is a vision model:
        if "vision" in model_name or "gpt-4o" in model_name or "o1" in model_name:
            return True

        # Any other model is not a vision model:
        return False

    def _has_audio_support(self, model_name: str) -> bool:

        if (
            "text-embedding" in model_name
            or "dall-e" in model_name
            or "gpt-3.5" in model_name
            or "gpt-4" in model_name
            or "realtime" in model_name
            or "o3-mini" in model_name
            or "o3-preview" in model_name
        ):
            return False

        # Check if the model is a whisper model:
        return "audio" in model_name or "whisper-1" in [
            model.id for model in self._raw_model_list
        ]

    def _has_video_support(self, model_name: str) -> bool:

        # First, we check if it is a text-embedding model or image generation model:
        if "text-embedding" in model_name or "dall-e" in model_name:
            return False

        # Then, we check if it is an audio model:
        if not (
            self._has_image_support(model_name)
            and self._has_text_gen_support(model_name)
        ) or not super().has_model_support_for(ModelFeatures.VideoConversion):
            return False

        return True

    def _has_document_support(self, model_name: str) -> bool:

        # Check if the model supports documents:
        if (
            not self._has_image_support(model_name)
            or not self._has_text_gen_support(model_name)
            or not super().has_model_support_for(ModelFeatures.DocumentConversion)
        ):
            # If the model does not support text gen and images, and there is no model in api that supports document conversion, we return False:
            return False

        return True

    def _has_structured_output_support(self, model_name: str) -> bool:
        # Check if the model supports structured output:
        if not self._has_text_gen_support(model_name):
            return False

        if (
            "dall-e" in model_name
            or "text-embedding" in model_name
            or "realtime" in model_name
            or "instruct" in model_name
            or "audio" in model_name
            or "chatgpt" in model_name
            or "gpt-3.5" in model_name
        ):
            return False

        if (
            "o1" in model_name
            or "o3" in model_name
            or "gpt-4o-mini" in model_name
            or "gpt-4o" in model_name
        ):
            return True

        if "gpt-3.5" in model_name or "gpt-4" in model_name:
            return False

        return True

    def _has_tools_support(self, model_name: str) -> bool:

        # Check if the model supports text gen which is a pre-requisite:
        if not self._has_text_gen_support(model_name):
            return False

        if (
            "gpt-3.5" in model_name
            or "dall-e" in model_name
            or "text-embedding" in model_name
            or "realtime" in model_name
            or "chatgpt" in model_name
            or "o1" in model_name
        ):
            return False

        return True

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Return the maximum context window (in tokens) for a given model,
        based on the current OpenAI model documentation.
        (Despite the function name, this is the total token capacity — input + output.)

        If model_name is None, this uses the default model name as a fallback.
        """
        if model_name is None:
            model_name = self.get_best_model()

        name = model_name.lower()

        # ============================
        # o1 family
        # ============================
        # o1: 200k token context
        # o1-mini: 128k token context
        # o1-preview: 128k token context

        if "o1-mini" in name:
            return 128000

        if "o1-preview" in name:
            return 128000

        if "o1" in name or "o3" in name:
            # e.g. "o1-2024-12-17"
            return 200000

        # ============================
        # GPT-4o family (omni)
        # ============================
        # GPT-4o, GPT-4o-mini, GPT-4o-audio, etc.
        # All standard GPT-4o versions: 128k token context

        if "gpt-4.5" in name:
            # gpt-4.5:
            return 128000

        if "gpt-4o-mini-realtime" in name:
            # Realtime previews do not change the total context limit
            # (still 128k context overall).
            return 128000

        if "gpt-4o-realtime" in name:
            return 128000

        if "gpt-4o-audio" in name:
            return 128000

        if "gpt-4o-mini" in name:
            return 128000

        if "chatgpt-4o-latest" in name:
            return 128000

        if "gpt-4o" in name:
            return 128000

        # ============================
        # GPT-4 Turbo
        # ============================
        # GPT-4 Turbo (and previews) have a 128k token context

        if "gpt-4-turbo" in name:
            return 128000

        # GPT-4 older previews sometimes end with “-preview”:
        # gpt-4-0125-preview, gpt-4-1106-preview => 128k tokens
        if "gpt-4-1106-preview" in name or "gpt-4-0125-preview" in name:
            return 128000

        # ============================
        # GPT-4 (Standard)
        # ============================
        # The standard GPT-4: 8,192 tokens

        if "gpt-4" in name:
            return 8192

        # ============================
        # GPT-3.5 Turbo
        # ============================
        # GPT-3.5 turbo-based models: 16,385 tokens

        if "gpt-3.5-turbo-instruct" in name:
            # Special case: gpt-3.5-turbo-instruct has a 4k context
            return 4096

        if "gpt-3.5-turbo" in name:
            return 16385

        # ============================
        # GPT base families
        # ============================
        # The newer GPT base models have a 16k context
        # e.g. babbage-002, davinci-002
        if "babbage-002" in name or "davinci-002" in name:
            return 16384

        # ============================
        # Default / Fallback
        # ============================
        # For any unknown model, return a safe 4k context
        return 4096

    from typing import Optional

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Return the model's 'Max output tokens' as defined in OpenAI's documentation.

        Note: This is distinct from the context window. The 'Max output tokens' refers
        to how many tokens can be generated by the model in its response (beyond inputs
        and any chain-of-thought tokens). The actual limit in practice may vary based
        on your request parameters (e.g., 'max_tokens', 'messages', etc.), but this
        function returns the documented cap for each model.

        If model_name is None, this uses get_default_openai_model_name() as a fallback.
        """

        if model_name is None:
            model_name = self.get_best_model()

        name = model_name.lower()

        # -----------------------------------
        # o1 and o1-mini
        # -----------------------------------
        #  • o1 => 200,000 tokens context, up to 100,000 for output
        #  • o1-mini => 128,000 context, up to 65,536 for output
        #  • o1-preview => 128,000 context, up to 32,768 for output
        if "o1-mini" in name:
            return 65536
        if "o1-preview" in name:
            return 32768
        if "o1" in name or "o3" in name:
            return 100000

        # -----------------------------------
        # GPT-4o (omni) family
        # -----------------------------------
        #  • gpt-4o => 128k context, 16,384 max output
        #  • gpt-4o-mini => 128k context, 16,384 max output
        #  • gpt-4o-realtime-preview => 128k context, 4,096 max output
        #  • gpt-4o-mini-realtime-preview => 128k context, 4,096 max output
        #  • gpt-4o-audio-preview => 128k context, 16,384 max output
        #  • chatgpt-4o-latest => 128k context, 16,384 max output
        if "gpt-4o-mini-realtime" in name:
            return 4096
        if "gpt-4o-realtime" in name:
            return 4096
        if "gpt-4o-audio" in name:
            return 16384
        if "gpt-4o-mini" in name:
            return 16384
        if "chatgpt-4o-latest" in name:
            return 16384
        if "gpt-4o" in name:
            return 16384

        # -----------------------------------
        # GPT-4 Turbo and older GPT-4 previews
        # -----------------------------------
        #  • gpt-4-turbo => 128k context, 4,096 max output
        #  • gpt-4-0125-preview, gpt-4-1106-preview => 128k context, 4,096 max output
        if "gpt-4-turbo" in name:
            return 4096
        if "gpt-4-1106-preview" in name or "gpt-4-0125-preview" in name:
            return 4096

        # -----------------------------------
        # GPT-4.5 family
        # -----------------------------------
        if "gpt-4.5" in name:
            # gpt-4.5:
            return 16384

        # -----------------------------------
        # GPT-4 (standard)
        # -----------------------------------
        #  • gpt-4 => 8,192 context, 8,192 max output
        if "gpt-4" in name:
            return 8192

        # -----------------------------------
        # GPT-3.5 Turbo family
        # -----------------------------------
        #  • gpt-3.5-turbo => 16,385 context, 4,096 max output
        #  • gpt-3.5-turbo-1106 => same
        #  • gpt-3.5-turbo-instruct => 4,096 context, 4,096 max output
        if "gpt-3.5-turbo-instruct" in name:
            return 4096
        if "gpt-3.5-turbo" in name:
            return 4096

        # -----------------------------------
        # Fallback
        # -----------------------------------
        # If a model isn't recognized, return a safe default (4k).
        return 4096

    def generate_text(
        self,
        messages: List[Message],
        model_name: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_num_output_tokens: Optional[int] = None,
        toolset: Optional[ToolSet] = None,
        use_tools: bool = True,
        response_format: Optional[BaseModel] = None,
        **kwargs,
    ) -> List[Message]:

        from openai import NotGiven

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
                raise APIError(f"No suitable model with features: {features}")

        if "o1" in model_name or "o3" in model_name:
            # ox models do not support any temperature except the default 1:
            temperature = 1.0

        # OpenAI specific preprocessing:
        preprocessed_messages = openai_preprocess_messages(
            model_name=model_name, messages=messages
        )

        # Preprocess messages:
        preprocessed_messages = self._preprocess_messages(
            messages=preprocessed_messages,
            deepcopy=False,  # We have already made a deepcopy
        )

        # Get max num of output tokens for model if not provided:
        if max_num_output_tokens is None:
            max_num_output_tokens = self.max_num_output_tokens(model_name)

        # Get max num of input tokens for model if not provided:
        max_num_input_tokens = self.max_num_input_tokens(model_name)

        # Estimate the number of tokens given a conversion of approx. 0.75 words per token:
        effective_max_output_tokens = estimate_num_input_tokens(
            max_num_input_tokens=max_num_input_tokens,
            max_num_output_tokens=max_num_output_tokens,
            preprocessed_messages=preprocessed_messages,
        )

        # Convert ToolSet to OpenAI-compatible JSON schema if toolset is provided
        openai_tools = format_tools_for_openai(toolset) if toolset else NotGiven()

        # Reasoning effort:
        if self._has_thinking_support(model_name):
            # get what is after the last '-' in the name:
            reasoning_effort = model_name.split("-")[-1]
            # Remove that postfix from the name:
            model_name = model_name.replace(f"-{reasoning_effort}", "")
        else:
            reasoning_effort = NotGiven()

        # Make sure that reponse_format is NotGiven if not set:
        if response_format is None:
            response_format = NotGiven()

        # List of tool_use blocks to append to the response message:
        new_messages = []

        try:
            # Loop until we get a response that doesn't require tool use:
            while True:

                # Format messages for OpenAI:
                openai_formatted_messages = convert_messages_for_openai(
                    preprocessed_messages
                )

                # Call OpenAI Chat Completions API
                with self.client.beta.chat.completions.stream(
                    model=model_name,
                    messages=openai_formatted_messages,
                    temperature=temperature,
                    max_completion_tokens=effective_max_output_tokens,
                    tools=openai_tools,
                    response_format=response_format,
                    reasoning_effort=reasoning_effort,
                    **kwargs,
                ) as streaming_response:
                    # Process the stream
                    for event in streaming_response:
                        # pprint(event)
                        if event.type == "content.delta":
                            self.callback_manager.on_text_streaming(
                                fragment=event.delta, **kwargs
                            )

                    # Get the final completion
                    openai_response = streaming_response.get_final_completion()

                # Process the response from OpenAI:
                response = process_response_from_openai(
                    openai_response=openai_response, response_format=response_format
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

                # Process tool calls:
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
            # print stack trace:
            import traceback

            traceback.print_exc()
            raise APIError(f"OpenAI generate text error: {e}")

        return new_messages

    def transcribe_audio(
        self, audio_uri: str, model_name: Optional[str] = None, **kwargs
    ) -> str:

        # If no model name is provided, use the best model:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.AudioTranscription)

        # If no model is available then we raise an error:
        if model_name is None:
            raise APIError("No model available for audio transcription.")

        # If the model is provided by the super class then delegate transcription to it:
        if model_name in super().list_models():
            return super().transcribe_audio(
                audio_uri=audio_uri, model_name=model_name, **kwargs
            )

        # We check that the model name contains 'whisper':
        if "whisper" not in model_name:
            raise APIError(f"Model {model_name} does not support audio transcription.")

        # Get the local path to the audio file:
        audio_file_path = uri_to_local_file_path(audio_uri)

        # Open the audio file:
        with open(audio_file_path, "rb") as audio_file:

            # Transcribe the audio file:
            transcription = self.client.audio.transcriptions.create(
                model=model_name, file=audio_file, response_format="text"
            )

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name})

        # Call the callback manager:
        self.callback_manager.on_audio_transcription(
            audio_uri=audio_uri, transcription=transcription, **kwargs
        )

        return transcription

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> str:

        # If no voice is provided, use the default voice:
        if voice is None:
            voice = "onyx"

        # If no file format is provided, use mp3:
        if audio_format is None:
            audio_format = "mp3"

        # Call the OpenAI API to generate audio:
        response = self.client.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            response_format=audio_format,
            input=text,
            **kwargs,
        )

        # Get the path of a temporary file to save the audio using tempfile, file should not be created:
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            # Save the audio to the temporary file:
            response.write_to_file(temp_file.name)

            # Return the URI to the temporary file:
            audio_file_uri = "file://" + temp_file.name

            # Update kwargs with other parameters:
            kwargs.update(
                {"voice": voice, "audio_format": audio_format, "model_name": model_name}
            )

            # Call the callback manager:
            self.callback_manager.on_audio_generation(
                text=text, audio_uri=audio_file_uri, **kwargs
            )

            return audio_file_uri

    def generate_image(
        self,
        positive_prompt: str,
        negative_prompt: Optional[str] = None,
        model_name: str = None,
        image_width: int = 512,
        image_height: int = 512,
        preserve_aspect_ratio: bool = True,
        allow_resizing: bool = True,
        **kwargs,
    ) -> "Image":

        # Make prompt from positive and negative prompts:
        if negative_prompt:
            prompt = f"Image must consist in: {positive_prompt}\nBut NOT of: {negative_prompt}"
        else:
            prompt = positive_prompt

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.ImageGeneration)

        # Allowed sizes by OpenAI:
        if "dall-e-2" in model_name:
            allowed_sizes = [
                "256x256",
                "512x512",
                "1024x1024",
                "1792x1024",
                "1024x1792",
            ]
        elif "dall-e-3" in model_name:
            allowed_sizes = ["1024x1024", "1024x1792", "1792x1024"]
        else:
            raise ValueError(f"Model {model_name} is not supported.")

        # Requested size:
        requested_size = f"{image_width}x{image_height}"

        # Check if requested size is allowed:
        if requested_size not in allowed_sizes:
            if not allow_resizing:
                raise ValueError(
                    f"Requested resolution {requested_size} is not allowed and resizing is not permitted."
                )

            # Find the closest higher resolution
            allowed_widths = sorted(
                set(int(size.split("x")[0]) for size in allowed_sizes)
            )
            allowed_heights = sorted(
                set(int(size.split("x")[1]) for size in allowed_sizes)
            )

            closest_width = next(
                (w for w in allowed_widths if w >= image_width), allowed_widths[-1]
            )
            closest_height = next(
                (h for h in allowed_heights if h >= image_height), allowed_heights[-1]
            )

            closest_size = f"{closest_width}x{closest_height}"
        else:
            closest_size = requested_size

        # Generate image:
        response = self.client.images.generate(
            model=model_name,
            prompt=prompt,
            size=closest_size,
            quality="hd",
            n=1,
            **kwargs,
        )

        # Get the image URL and download the image:
        image_url = response.data[0].url
        image = Image.open(BytesIO(requests.get(image_url).content))

        # Resize the image if needed:
        if closest_size != requested_size:
            if preserve_aspect_ratio:
                # Preserve aspect ratio:
                image.thumbnail((image_width, image_height), Resampling.LANCZOS)
            else:
                # Resize without preserving aspect ratio:
                image = image.resize((image_width, image_height), Resampling.LANCZOS)

        # Update kwargs with other parameters:
        kwargs.update(
            {
                "model_name": model_name,
                "negative_prompt": negative_prompt,
                "image_width": image_width,
                "image_height": image_height,
                "preserve_aspect_ratio": preserve_aspect_ratio,
                "allow_resizing": allow_resizing,
            }
        )

        # Call the callback manager:
        self.callback_manager.on_image_generation(image=image, prompt=prompt, **kwargs)

        return image

    def embed_texts(
        self,
        texts: Sequence[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.TextEmbeddings)

        # If the model belongs to the superclass, we call the superclass method:
        if model_name in super().list_models():
            return super().embed_texts(
                texts=texts, model_name=model_name, dimensions=dimensions, **kwargs
            )

        # Call OpenAI API to generate embeddings:
        response = self.client.embeddings.create(
            model=model_name, dimensions=dimensions, input=texts, **kwargs
        )

        # Extract embeddings from the response:
        embeddings = list([e.embedding for e in response.data])

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager:
        self.callback_manager.on_text_embedding(
            texts=texts, embeddings=embeddings, **kwargs
        )

        return embeddings
