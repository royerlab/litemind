import os
from typing import List, Optional, Sequence, Set, Type, Union

from PIL import Image as PilImage
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.feature_scanner import get_default_model_feature_scanner
from litemind.apis.providers.google.utils.aggegate_chat_response import (
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
from litemind.apis.providers.google.utils.response_to_object import response_to_object
from litemind.apis.tests.test_callback_manager import callback_manager
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_text import Text
from litemind.utils.json_to_object import json_to_object


class GeminiApi(DefaultApi):
    """
    A Gemini 1.5+ API implementation conforming to the BaseApi interface.
    Uses the google.generativeai library (previously known as Google GenAI).

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
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ):
        """
        Initialize the Gemini client.

        Parameters
        ----------
        api_key : Optional[str]
            The API key for Google GenAI. If not provided, reads from GOOGLE_API_KEY.
        allow_media_conversions: bool
            If True, the API will allow media conversions using the default media converter.
        allow_media_conversions_with_models: bool
            If True, the API will allow media conversions using models that support the required features in addition to the default media converter.
            To use this the allow_media_conversions parameter must be True.
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
            # Initialize the feature scanner:
            self.feature_scanner = get_default_model_feature_scanner()

            # google.generativeai references
            import google.generativeai as genai

            # Register the API key with google.generativeai
            genai.configure(api_key=self._api_key, **self.kwargs)

            # Get the raw model list:
            self._model_list = _get_gemini_models_list()

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing Gemini client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:

        # Check the availability of the API:
        result = check_gemini_api_availability(api_key=api_key)

        # Call the callback manager:
        self.callback_manager.on_availability_check(result)

        # Ensure that the correct key is set:
        import google.generativeai as genai

        genai.configure(api_key=self._api_key, **self.kwargs)

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
            if not self.feature_scanner.supports_feature(
                self.__class__, model_name, feature
            ):
                # If the model does not support the feature, we return False:
                return False

        return True

    def _has_thinking_support(self, model_name: str) -> bool:
        if (
            "models/gemini" not in model_name.lower()
            and "thinking" not in model_name.lower()
        ):
            return False
        return True

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model()

        # Normalise the model name to lower case:
        name_lower = model_name.lower()

        from google.generativeai import (
            list_models,
        )  # This is the function from your snippet

        for model_obj in list_models():
            # model_obj is a Model protobuf (or typed dict) with a .name attribute
            # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
            if name_lower == model_obj.name.lower():
                return model_obj.input_token_limit

        # If we reach this point then the model was not found!

        # So we call super class method:
        return super().max_num_input_tokens(model_name=model_name)

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model()

        # Normalise the model name to lower case:
        name_lower = model_name.lower()

        from google.generativeai import (
            list_models,
        )  # This is the function from your snippet

        for model_obj in list_models():
            # model_obj is a Model protobuf (or typed dict) with a .name attribute
            # e.g. "models/gemini-1.5-flash", "models/gemini-2.0-flash-exp", etc.
            if name_lower == model_obj.name.lower():
                return model_obj.output_token_limit

        # If we reach this point then the model was not found!

        # So we call super class method:
        return super().max_num_output_tokens(model_name=model_name)

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

        import google.generativeai as genai
        from google.generativeai import types

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

        # If reasoning model is selected, we request that the thinking be enclosed in a <thinking> ... <thinking/ tag:
        if self._has_thinking_support(model_name):
            system_instruction += "\nThink carefully step-by-step before responding: restate the input, analyze it, consider options, make a plan, and proceed methodically to your conclusion. \n"
            system_instruction += (
                f"All reasoning (thinking) which precedes the final answer must be enclosed within thinking tags."
                f"This is how your response should be formatted: <thinking> reasoning goes here... </thinking> final answer goes here...\n\n"
            )

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

        # Build GenerationConfig
        if response_format is None or toolset is not None:
            generation_cfg = types.GenerationConfig(
                temperature=temperature, max_output_tokens=max_num_output_tokens
            )
        else:
            generation_cfg = types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_num_output_tokens,
                response_mime_type="application/json",
                response_schema=response_format,
            )

        # Format tools for Gemini:
        gemini_tools = format_tools_for_gemini(toolset)

        # List of new messages part of the response:
        new_messages = []

        try:

            # Get model by name and set tools and config:
            model = genai.GenerativeModel(
                model_name=model_name,
                tools=gemini_tools,
                generation_config=generation_cfg,
                system_instruction=system_instruction,
            )

            # Start chat
            chat = model.start_chat()

            # Loop until we get a response that doesn't require tool use:
            while True:

                # Convert messages to gemini format:
                gemini_messages = convert_messages_for_gemini(preprocessed_messages)

                # Send messages to Gemini:
                streaming_response = chat.send_message(gemini_messages, stream=True)

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
                # preprocessed_messages.append(response)
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
                    set_preprocessed=True,
                )

            if response_format:

                # Prepare the messages for formatting:
                formatting_messages = messages.copy()

                # Add a message to the user to convert the response to JSON:
                message = Message(
                    role="user",
                    text="Convert the answer above to JSON adhering to the following schema:\n{response_format.model_json_schema()}\n",
                )
                formatting_messages.append(message)

                # Generate the prompt for the user:
                structured_message = response_to_object(
                    messages=formatting_messages,
                    model_name=model_name,
                    max_num_output_tokens=max_num_output_tokens,
                    response_format=response_format,
                )
                # If the message is not None or empty, set the processed response to the structured message:
                if structured_message:
                    json_to_object(
                        response, response_format, structured_message[0].get_content()
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

        import google.generativeai as genai

        imagen = genai.ImageGenerationModel(model_id=model_name)

        # Computes aspect ratio from resolution:
        aspect_ratio = image_width / image_height

        # Supported values are: "1:1", "3:4", "4:3", "9:16", and "16:9", snap aspect_ratio to closest:
        if aspect_ratio == 1:
            aspect_ratio = "1:1"
        elif aspect_ratio == 0.75:
            aspect_ratio = "3:4"
        elif aspect_ratio == 1.33:
            aspect_ratio = "4:3"
        elif aspect_ratio == 0.56:
            aspect_ratio = "9:16"
        elif aspect_ratio == 1.77:
            aspect_ratio = "16:9"

        result = imagen.generate_images(
            prompt=positive_prompt,
            number_of_images=1,
            safety_filter_level="block_only_high",
            person_generation="allow_adult",
            aspect_ratio=aspect_ratio,
            negative_prompt=negative_prompt,
        )

        # Get the generated image:
        generated_image = result[0].image

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

        # Local import to avoid loading the library if not needed:
        import google.generativeai as genai

        # Generate the embeddings:
        embeddings = []

        for text in texts:
            # Generate the embeddings:
            result = genai.embed_content(
                model=model_name, content=text, output_dimensionality=dimensions
            )

            # Get the embeddings:
            embedding = result["embedding"]

            # Get the embeddings:
            embeddings.append(embedding)

        # Add other parameters to kwargs dict:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager:
        self.callback_manager.on_text_embedding(
            texts=texts, embeddings=embeddings, **kwargs
        )

        return embeddings
