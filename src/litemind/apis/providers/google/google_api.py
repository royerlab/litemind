import os
from typing import List, Optional, Sequence, Union

from PIL import Image
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.providers.google.utils.aggegate_chat_response import (
    aggregate_chat_response,
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
from litemind.apis.utils.json_to_object import json_to_object


class GeminiApi(DefaultApi):
    """
    A Gemini 1.5+ API implementation conforming to the BaseApi interface.
    Uses the google.generativeai library (previously known as Google GenAI).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ):
        """
        Initialize the Gemini client.

        Parameters
        ----------
        api_key : Optional[str]
            The API key for Google GenAI. If not provided, reads from GOOGLE_API_KEY.
        kwargs : dict
            Additional parameters (unused here, but accepted for consistency).
        """

        super().__init__(callback_manager=callback_manager)

        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise APIError(
                "A valid GOOGLE_API_KEY is required for GeminiApi. "
                "Set GOOGLE_API_KEY in the environment or pass api_key explicitly."
            )

        self._api_key = api_key

        try:
            # google.generativeai references
            import google.generativeai as genai

            # Register the API key with google.generativeai
            genai.configure(api_key=self._api_key, **kwargs)

            # Get the raw model list:
            self._model_list = _get_gemini_models_list()

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing Gemini client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:

        # Local import to avoid loading the library if not needed:
        import google.generativeai as genai

        # Use the provided key or the default key:
        candidate_key = api_key or self._api_key
        if not candidate_key:
            self.callback_manager.on_availability_check(False)
            return False

        # We’ll try a trivial call; if it fails, we assume invalid.
        try:
            # Minimal call: generate a short response
            model = genai.GenerativeModel()

            # Generate a short response:
            resp = model.generate_content("Hello, Gemini!")

            # If the response is not empty, we assume the API is available:
            if len(resp.text) > 0:
                self.callback_manager.on_availability_check(True)
                return True

            # If the response is empty, we assume the API is not available:
            self.callback_manager.on_availability_check(False)
            return False
        except Exception:
            import traceback

            traceback.print_exc()
            self.callback_manager.on_availability_check(False)
            return False

    def list_models(
        self, features: Optional[Sequence[ModelFeatures]] = None
    ) -> List[str]:

        try:

            # Get the full list of models:
            model_list = list(self._model_list)

            # Add the models from the super class:
            model_list += super().list_models()

            # To be safe, we remove duplicates but keep the order:
            model_list = list(dict.fromkeys(model_list))

            # Filter the models based on the features:
            if features:
                model_list = self._filter_models(model_list, features=features)

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
        exclusion_filters: Optional[Union[str, List[str]]] = None,
    ) -> Optional[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the full list of models:
        model_list = self.list_models()

        # Filter the models based on the requirements:
        model_list = self._filter_models(
            model_list, features=features, exclusion_filters=exclusion_filters
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
                if "models/gemini" not in model_name.lower():
                    return False

            elif feature == ModelFeatures.ImageGeneration:
                # FIXME: We need to figure out how to call the video generation API, not working right now.
                return False
                # if 'imagen' not in model_name.lower():
                #     return False

            elif feature == ModelFeatures.Reasoning:
                if (
                    "models/gemini" not in model_name.lower()
                    or "thinking" in model_name.lower()
                ):
                    return False

            elif feature == ModelFeatures.TextEmbeddings:
                if "text-embedding" not in model_name.lower():
                    return False

            elif feature == ModelFeatures.ImageEmbeddings:
                if "text-embedding" not in model_name.lower():
                    return False

            elif feature == ModelFeatures.AudioEmbeddings:
                if "text-embedding" not in model_name.lower():
                    return False

            elif feature == ModelFeatures.VideoEmbeddings:
                if "text-embedding" not in model_name.lower():
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

    def _has_image_support(self, model_name: Optional[str] = None) -> bool:

        if not model_name:
            model_name = self.get_best_model()
        return "gemini-1.5" in model_name.lower() or "gemini-2.0" in model_name.lower()

    def _has_audio_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.get_best_model()
        # Current assumption: Gemini models support audio if they are multimodal:
        return self._has_image_support(model_name=model_name)

    def _has_video_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.get_best_model()
        # Current assumption: Gemini models support video if they are multimodal:
        return self._has_image_support(model_name) and "thinking" not in model_name

    def _has_document_support(self, model_name: Optional[str] = None) -> bool:
        return self.has_model_support_for(
            [ModelFeatures.Image, ModelFeatures.TextGeneration], model_name
        ) and self.has_model_support_for(ModelFeatures.DocumentConversion)

    def _has_tool_support(self, model_name: Optional[str] = None) -> bool:

        if not model_name:
            model_name = self.get_best_model()

        model_name = model_name.lower()

        # Experimental models are tricky and typically don't support tools:
        if "exp" in model_name:
            return False

        # Check for specific models that support tools:
        return (
            "gemini-1.5-flash" in model_name
            or "gemini-1.5-pro" in model_name
            or "gemini-1.0-pro" in model_name
            or "gemini-2.0" in model_name
        )

    def _has_structured_output_support(self, model_name: Optional[str] = None) -> bool:
        if not model_name:
            model_name = self.get_best_model()

        model_name = model_name.lower()

        # Experimental models are tricky and typically don't support structured output:
        if "exp" in model_name:
            return False

        # Check for specific models that support structured output:
        return (
            "gemini-1.5-flash" in model_name
            or "gemini-1.5-pro" in model_name
            or "gemini-1.0-pro" in model_name
            or "gemini-2.0" in model_name
        )

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

        # Convert user messages -> gemini format
        preprocessed_messages = self._preprocess_messages(
            messages=messages, convert_videos=False
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
                    model_name=model_name,
                    max_num_output_tokens=max_num_output_tokens,
                    response_format=response_format,
                )

                # Append response message to original, preprocessed, and new messages:
                messages.append(response)
                # preprocessed_messages.append(response)
                new_messages.append(response)

                # If the model wants to use a tool, parse out the tool calls:
                if not response.has(BlockType.Tool):
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
                        response, response_format, structured_message[0].content
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
    ) -> Image:

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
        texts: List[str],
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
