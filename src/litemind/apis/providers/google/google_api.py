import os
from typing import List, Optional, Sequence, Set, Type, Union

from PIL import Image as PilImage
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.api_callback_manager import ApiCallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.model_registry import get_default_model_registry
from litemind.apis.providers.google.utils.aggregate_chat_response import (
    aggregate_chat_response,
)
from litemind.apis.providers.google.utils.check_availability import (
    check_gemini_api_availability,
)
from litemind.apis.providers.google.utils.convert_messages import (
    convert_messages_for_gemini,
)
from litemind.apis.providers.google.utils.format_tools import format_tools_for_gemini
from litemind.apis.providers.google.utils.list_models import _get_gemini_models_list
from litemind.apis.providers.google.utils.process_response import (
    process_response_from_gemini,
)
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_text import Text

# Default token limits for unknown models
_DEFAULT_CONTEXT_WINDOW = 1_000_000
_DEFAULT_MAX_OUTPUT_TOKENS = 65_536

# Supported aspect ratios for Gemini image generation
_SUPPORTED_ASPECT_RATIOS = {
    1.0: "1:1",
    0.75: "3:4",
    1.333: "4:3",
    0.5625: "9:16",
    1.778: "16:9",
}


def _snap_to_aspect_ratio(aspect_ratio: float) -> str:
    """Snap to closest supported Gemini aspect ratio."""
    closest = min(_SUPPORTED_ASPECT_RATIOS.keys(), key=lambda r: abs(r - aspect_ratio))
    return _SUPPORTED_ASPECT_RATIOS[closest]


class GeminiApi(DefaultApi):
    """
    A Gemini 1.5+ API implementation conforming to the BaseApi interface.
    Uses the google.genai library (unified Google GenAI SDK).

    Gemini models support text generation, image inputs, audio and video inputs, and thinking,
    but do not support all features natively. Support for these features
    are provided by the Litemind API via our fallback mechanism.

    Set the GOOGLE_GEMINI_API_KEY environment variable to your Google API key.
    You can get a key from the Google Cloud Console: https://ai.google.dev/gemini-api/docs/api-key

    Note: The API key must be enabled for the Gemini API.
    You can also pass the API key explicitly to the constructor.

    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        allow_media_conversions: bool = True,
        allow_media_conversions_with_models: bool = True,
        callback_manager: Optional[ApiCallbackManager] = None,
        **kwargs,
    ):
        """
        Initialize the Gemini client.

        Parameters
        ----------
        api_key : Optional[str]
            The API key for Google GenAI. If not provided, reads from GOOGLE_GEMINI_API_KEY.
        allow_media_conversions: bool
            If True, the API will allow media conversions using the default media converter.
        allow_media_conversions_with_models: bool
            If True, the API will allow media conversions using models that support
            the required features in addition to the default media converter.
            To use this, the allow_media_conversions parameter must be True.
        callback_manager : Optional[ApiCallbackManager]
            Callback manager for API events.
        kwargs : dict
            Additional parameters (unused here, but accepted for consistency).
        """

        super().__init__(
            allow_media_conversions=allow_media_conversions,
            allow_media_conversions_with_models=allow_media_conversions_with_models,
            callback_manager=callback_manager,
        )

        if api_key is None:
            api_key = os.environ.get("GOOGLE_GEMINI_API_KEY")
        if not api_key:
            raise APIError(
                "A valid GOOGLE_GEMINI_API_KEY is required for GeminiApi. "
                "Set GOOGLE_GEMINI_API_KEY in the environment or pass api_key explicitly."
            )

        self._api_key = api_key
        self.kwargs = kwargs

        try:
            # Initialize the model registry for feature lookups:
            self.model_registry = get_default_model_registry()

            # google.genai references - new unified SDK
            from google.genai import Client

            # Create the client instance (replaces global genai.configure)
            # Note: Using default timeout as custom timeouts can cause issues
            self.client = Client(api_key=self._api_key)

            # Get the raw model list:
            self._model_list = _get_gemini_models_list(self.client)

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing Gemini client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:

        # Check the availability of the API using the client:
        result = check_gemini_api_availability(self.client)

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
        media_types: Optional[Set[Type[MediaBase]]] = None,
    ) -> List[str]:

        try:
            # Normalise the features:
            features = ModelFeatures.normalise(features)
            non_features = ModelFeatures.normalise(non_features)

            # Get the full list of models:
            model_list = list(self._model_list)

            # Add the models from the super class:
            model_list += super().list_models()

            # To be safe, we remove duplicates but keep the order:
            model_list = list(dict.fromkeys(model_list))

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
            raise APIError("Error fetching model list from Google.")

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

        # Get the full list of models:
        model_list = self.list_models()

        # Filter the models based on the requirements:
        model_list = self._filter_models(
            model_list,
            features=features,
            non_features=non_features,
            media_types=media_types,
            exclusion_filters=exclusion_filters,
        )

        if model_list:
            # If we have any models left, return the first one:
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

        for feature in features:
            if not self.model_registry.supports_feature(
                self.__class__, model_name, feature
            ):
                # If the model does not support the feature, we return False:
                return False

        return True

    def _has_thinking_support(self, model_name: str) -> bool:
        """Check if model supports thinking via registry."""
        return self.model_registry.supports_feature(
            self.__class__, model_name, ModelFeatures.Thinking
        )

    def _build_thinking_config(self, model_name: str, max_output_tokens: int):
        """
        Build a ThinkingConfig for models that support native thinking.

        Parameters
        ----------
        model_name : str
            The model name to check for thinking support.
        max_output_tokens : int
            Maximum output tokens to compute thinking budget.

        Returns
        -------
        ThinkingConfig or None
            ThinkingConfig if the model supports thinking, None otherwise.
        """
        if not self._has_thinking_support(model_name):
            return None

        from google.genai import types

        # Compute thinking budget as a fraction of max output tokens
        thinking_budget = max(1024, max_output_tokens // 3)

        model_lower = model_name.lower()

        # Gemini 3.x preview models don't fully support thinking_level yet
        # Only apply thinking for non-preview gemini-3 models or explicit thinking models
        if "gemini-3" in model_lower:
            # Skip thinking for preview models as they may not support all thinking levels
            if "preview" in model_lower:
                return None
            return types.ThinkingConfig(thinking_level="medium")
        else:
            # Gemini 2.5 and other thinking models
            return types.ThinkingConfig(thinking_budget=thinking_budget)

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        if model_name is None:
            model_name = self.get_best_model()
        model_info = self.model_registry.get_model_info(self.__class__, model_name)
        if model_info and model_info.context_window:
            return model_info.context_window
        return _DEFAULT_CONTEXT_WINDOW

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        if model_name is None:
            model_name = self.get_best_model()
        model_info = self.model_registry.get_model_info(self.__class__, model_name)
        if model_info and model_info.max_output_tokens:
            return model_info.max_output_tokens
        return _DEFAULT_MAX_OUTPUT_TOKENS

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

        from google.genai import types

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

        # Set default model if not provided
        if model_name is None:
            model_name = self._get_best_model_for_text_generation(
                messages, toolset if use_tools else None, response_format
            )

        # Get system instruction from messages:
        system_instruction = ""
        for message in messages:
            if message.role == "system":
                for block in message.blocks:
                    if block.has_type(Text):
                        system_instruction += block.get_content()

        # Convert user messages -> gemini format
        preprocessed_messages = self._preprocess_messages(
            messages=messages,
            allowed_media_types=self._get_allowed_media_types_for_text_generation(
                model_name=model_name
            ),
        )

        # Get max num of output tokens for model if not provided:
        if max_num_output_tokens is None:
            max_num_output_tokens = self.max_num_output_tokens(model_name)

        # Build native ThinkingConfig if model supports it
        thinking_config = self._build_thinking_config(model_name, max_num_output_tokens)

        # Format tools for Gemini:
        gemini_tools = format_tools_for_gemini(toolset)

        # Build GenerationConfig (tools are part of the config in new SDK)
        if response_format is None or toolset is not None:
            generation_cfg = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_num_output_tokens,
                system_instruction=system_instruction if system_instruction else None,
                thinking_config=thinking_config,
                tools=gemini_tools,
            )
        else:
            generation_cfg = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_num_output_tokens,
                response_mime_type="application/json",
                response_schema=response_format,
                system_instruction=system_instruction if system_instruction else None,
                thinking_config=thinking_config,
                tools=gemini_tools,
            )

        # List of new messages part of the response:
        new_messages = []

        try:

            # Loop until we get a response that doesn't require tool use:
            while True:

                # Convert messages to gemini format:
                gemini_messages = convert_messages_for_gemini(
                    preprocessed_messages, self.client
                )

                # Send messages to Gemini using streaming:
                streaming_response = self.client.models.generate_content_stream(
                    model=model_name,
                    contents=gemini_messages,
                    config=generation_cfg,
                )

                # Aggregate the response:
                gemini_response = aggregate_chat_response(
                    streaming_response, self.callback_manager.on_text_streaming
                )

                # Process the response:
                response = process_response_from_gemini(
                    gemini_response,
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

                # Process the tool call
                self._process_tool_calls(
                    response,
                    messages,
                    new_messages,
                    preprocessed_messages,
                    toolset,
                    set_preprocessed=False,
                )

            # Fire final callback:
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
                response=response, messages=messages, **kwargs
            )

        except Exception as e:
            raise APIError(f"Gemini generate text error: {e}")

        return new_messages

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
    ) -> PilImage:

        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.ImageGeneration)

        from google.genai import types

        # Computes aspect ratio from resolution and snap to closest supported:
        aspect_ratio = image_width / image_height
        aspect_ratio_str = _snap_to_aspect_ratio(aspect_ratio)

        # Build the generation config
        # Note: Imagen models only support "BLOCK_LOW_AND_ABOVE" safety filter level
        config = types.GenerateImagesConfig(
            number_of_images=1,
            safety_filter_level="BLOCK_LOW_AND_ABOVE",
            person_generation="ALLOW_ADULT",
            aspect_ratio=aspect_ratio_str,
            negative_prompt=negative_prompt,
        )

        # Generate images using the client
        result = self.client.models.generate_images(
            model=model_name,
            prompt=positive_prompt,
            config=config,
        )

        # Get the generated image:
        generated_image = result.generated_images[0].image

        # Add all parameters to kwargs:
        kwargs.update(
            {
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "image_width": image_width,
                "image_height": image_height,
                "preserve_aspect_ratio": preserve_aspect_ratio,
                "allow_resizing": allow_resizing,
            }
        )

        # Call the callback manager:
        self.callback_manager.on_image_generation(
            prompt=positive_prompt, image=generated_image, **kwargs
        )

        return generated_image

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

        # If model belongs to super class delegate to it:
        if model_name in super().list_models():
            return super().embed_texts(
                texts=texts, model_name=model_name, dimensions=dimensions, **kwargs
            )

        from google.genai import types

        # Generate the embeddings:
        embeddings = []

        for text in texts:
            # Build embedding config
            config = types.EmbedContentConfig(
                output_dimensionality=dimensions,
            )

            # Generate the embeddings using the client:
            result = self.client.models.embed_content(
                model=model_name,
                contents=text,
                config=config,
            )

            # Get the embeddings:
            embedding = result.embeddings[0].values

            # Get the embeddings:
            embeddings.append(embedding)

        # Add other parameters to kwargs dict:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager:
        self.callback_manager.on_text_embedding(
            texts=texts, embeddings=embeddings, **kwargs
        )

        return embeddings
