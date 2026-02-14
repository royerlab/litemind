"""Unified API that aggregates multiple provider APIs with automatic fallback.

This module provides ``CombinedApi``, which wraps multiple LLM provider APIs
(OpenAI, Anthropic, Gemini, Ollama) behind a single ``DefaultApi`` interface.
Each call is delegated to the first available provider that supports the
required features. Providers are validated for availability and credentials
during initialization, and a model-to-API mapping allows transparent routing
of requests.
"""

from typing import Dict, List, Optional, Sequence, Type, Union

from arbol import aprint
from PIL.Image import Image
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.callbacks.api_callback_manager import ApiCallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError
from litemind.apis.model_features import ModelFeatures
from litemind.media.media_base import MediaBase


class CombinedApi(DefaultApi):
    """Unified API that aggregates multiple provider APIs with automatic fallback.

    CombinedApi wraps multiple API providers (OpenAI, Anthropic, Gemini, Ollama)
    behind a single interface. When a method is called, it delegates to the first
    available provider that supports the required features. Each provider is
    validated for availability and credentials during initialization.
    """

    def __init__(
        self,
        apis: Optional[List[BaseApi]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        callback_manager: Optional[ApiCallbackManager] = None,
    ):
        """Create a new combined API.

        If no APIs are provided, the default APIs are used: OpenAIApi,
        AnthropicApi, OllamaApi, GeminiApi. Only APIs that pass availability
        checks and have at least one model are included.

        Parameters
        ----------
        apis : Optional[List[BaseApi]]
            Specific API instances to combine. If None, uses all available
            default providers.
        api_keys : Optional[Dict[str, str]]
            API keys keyed by class name (e.g., ``{"OpenAIApi": "sk-..."}``).
        callback_manager : Optional[ApiCallbackManager]
            Callback manager propagated to all child APIs.
        """

        # call the parent constructor:
        super().__init__(callback_manager=callback_manager)

        # If no APIs are provided, use the default ones:
        if apis is None:

            # Import the APIs:
            from litemind.apis.providers.anthropic.anthropic_api import AnthropicApi
            from litemind.apis.providers.google.google_api import GeminiApi
            from litemind.apis.providers.ollama.ollama_api import OllamaApi
            from litemind.apis.providers.openai.openai_api import OpenAIApi

            # List of classes:
            potential_api_classes = [OpenAIApi, AnthropicApi, OllamaApi, GeminiApi]

            # List of classes that will be used to instantiate the APIs:
            self.api_classes = []

            # List of APIs to combine:
            apis = []

            # Add the APIs to the list:
            for api_class in potential_api_classes:
                try:
                    api_instance = api_class()
                    if api_instance.check_availability_and_credentials():

                        if len(api_instance.list_models()) > 0:
                            self.api_classes.append(api_class)
                            apis.append(api_instance)
                        else:
                            aprint(
                                f"API {api_class.__name__} does not provide any models. Removing it from the list."
                            )
                    else:
                        aprint(
                            f"API {api_class.__name__} is not available. Removing it from the list."
                        )
                except Exception as e:
                    aprint(
                        f"API {api_class.__name__} could not be instantiated: {e}. Removing it from the list."
                    )

        # Add the _callbacks from this CombinedApi class to the callback manager of each individual APIs:
        if callback_manager:
            for api in apis:
                api.callback_manager.add_callbacks(callback_manager)

        # Initialise the APIs lists and maps:
        self.apis = []
        self.model_to_api = {}

        # Check the availability of each API and add it to the list:
        for api in apis:

            # get the class name and api key:
            class_name = api.__class__.__name__

            # get the api key from the api_keys dictionary:
            api_key = (
                None
                if not api_keys or class_name not in api_keys
                else api_keys.get(class_name)
            )

            # Check the availability and credentials of the API:
            if not api.check_availability_and_credentials(api_key=api_key):
                aprint(
                    f"API {class_name} not added due to unavailability or invalid credentials."
                )
                continue

            # Add the API to the list and map:
            try:
                # get the models of the API:
                models = api.list_models()

                # Add the API to the list and map:
                for model in models:
                    # add the model to the map:
                    if model not in self.model_to_api:
                        self.model_to_api[model] = api
                self.apis.append(api)
            except Exception as e:
                aprint(f"API {api.__class__.__name__} not added due to error: {e}")

        # If no API is available, raise an error:
        if not self.apis:
            raise ValueError("No API available.")

    def check_availability_and_credentials(
        self, api_key: Optional[str] = None
    ) -> Optional[bool]:
        """Check whether any combined API has available models.

        Parameters
        ----------
        api_key : Optional[str]
            Ignored; availability is determined by the presence of
            registered models.

        Returns
        -------
        Optional[bool]
            True if at least one model is registered across all APIs.
        """
        # By definition, if we have added a model it is available.
        result = len(self.model_to_api) > 0

        # Call the callback manager if available:
        self.callback_manager.on_availability_check(result)

        # return the result:
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
        """List all models across combined APIs, optionally filtered by features.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required features to filter models by.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features to exclude from the results.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the returned models must support.

        Returns
        -------
        List[str]
            List of matching model names from all combined APIs.

        Raises
        ------
        APIError
            If an error occurs while fetching the model list.
        """
        try:
            # Normalise the features:
            features = ModelFeatures.normalise(features)
            non_features = ModelFeatures.normalise(non_features)

            # Get the list of models from the model_to_api dictionary:
            model_list = list(self.model_to_api.keys())

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

            # return the model list:
            return model_list

        except Exception as e:
            raise APIError(f"Error fetching model list from CombinedApi: {e}") from e

    def get_best_model(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        exclusion_filters: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Select the best model from any combined API matching the criteria.

        Iterates through provider APIs in priority order and returns the first
        model satisfying all feature, non-feature, media type, and exclusion
        constraints.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required features the model must support.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features the model must not have.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must be able to process.
        exclusion_filters : Optional[List[str]]
            Substrings that, if found in a model name, exclude it.

        Returns
        -------
        Optional[str]
            The best matching model name, or None if no model matches.
        """
        for api in self.apis:
            try:

                # get the best model from the api object:
                best_model = api.get_best_model(
                    features=features,
                    non_features=non_features,
                    media_types=media_types,
                    exclusion_filters=exclusion_filters,
                )

                # if the best model is None or not in the model_to_api dictionary we continue:
                if best_model is None or best_model not in self.model_to_api:
                    continue

                # The first model to reach this point 'wins'

                # Set kwargs to other parameters:
                kwargs = {"features": features, "exclusion_filters": exclusion_filters}

                # Call the callback manager:
                self.callback_manager.on_best_model_selected(best_model, **kwargs)

                # return the best model:
                return best_model
            except Exception as e:
                aprint(f"Error getting best model from {api.__class__.__name__}: {e}")

        # If no model is found, return None:
        return None

    def has_model_support_for(
        self,
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]],
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        model_name: Optional[str] = None,
    ) -> bool:
        """Check if a model supports the given features and media types.

        Delegates to the provider API that owns the model.

        Parameters
        ----------
        features : Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            The features to check for.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must support.
        model_name : Optional[str]
            The model to check. If None, checks any available model.

        Returns
        -------
        bool
            True if the model supports the requested features and media types.

        Raises
        ------
        ValueError
            If the model name is not found in any combined API.
        """
        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # if no model name is provided we return the best model as defined in the method above:
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

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the has_model_support_for method of the api object,
        # and check if the model supports the features:
        has_support = api.has_model_support_for(
            features=features, media_types=media_types, model_name=model_name
        )

        # return the result:
        return has_support

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of input tokens for a model.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query. If None, uses the best available model.

        Returns
        -------
        int
            The maximum number of input tokens.

        Raises
        ------
        ValueError
            If the model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model()

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the max_num_input_tokens method of the api object:
        return api.max_num_input_tokens(model_name=model_name)

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of output tokens for a model.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query. If None, uses the best available model.

        Returns
        -------
        int
            The maximum number of output tokens.

        Raises
        ------
        ValueError
            If the model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model()

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the max_num_output_tokens method of the api object:
        return api.max_num_output_tokens(model_name=model_name)

    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """Count the number of tokens in the given text using the model's tokenizer.

        Parameters
        ----------
        text : str
            The text to tokenize and count.
        model_name : Optional[str]
            The model whose tokenizer to use. If None, uses the best
            available model.

        Returns
        -------
        int
            The number of tokens.

        Raises
        ------
        ValueError
            If the model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model()

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the count_tokens method of the api object:
        return api.count_tokens(text, model_name)

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
        """Generate text by delegating to the appropriate provider API.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to send to the model.
        model_name : Optional[str]
            The model to use. If None, selects the best text generation model.
        temperature : float
            Sampling temperature (0.0 = deterministic).
        max_num_output_tokens : Optional[int]
            Maximum tokens in the response.
        toolset : Optional[ToolSet]
            The toolset available for function calling.
        use_tools : bool
            If True, executes tool calls automatically.
        response_format : Optional[BaseModel]
            Pydantic model for structured output parsing.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        List[Message]
            Response messages from the provider.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.TextGeneration)

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the generate_text_completion method of the api object:
        response = api.generate_text(
            messages=messages,
            model_name=model_name,
            temperature=temperature,
            max_num_output_tokens=max_num_output_tokens,
            toolset=toolset,
            use_tools=use_tools,
            response_format=response_format,
        )

        # Call the callback manager if available:
        self.callback_manager.on_text_generation(messages, response=response, **kwargs)

        # return the result:
        return response

    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Generate audio by delegating to the appropriate provider API.

        Parameters
        ----------
        text : str
            The text to convert to audio.
        voice : Optional[str]
            The voice to use for synthesis.
        audio_format : Optional[str]
            Audio file format (e.g., "mp3", "wav").
        model_name : Optional[str]
            The model to use. If None, selects the best audio generation model.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        str
            URI of the generated audio file.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.AudioGeneration)

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the generate_audio method of the api object:
        audio_uri = api.generate_audio(text, voice, audio_format, model_name)

        # Update kwargs with other parameters:
        kwargs.update(
            {"voice": voice, "audio_format": audio_format, "model_name": model_name}
        )

        # Call the callback manager if available:
        self.callback_manager.on_audio_generation(text, audio_uri, **kwargs)

        # return the result:
        return audio_uri

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
        """Generate an image by delegating to the appropriate provider API.

        Parameters
        ----------
        positive_prompt : str
            The prompt describing what to generate.
        negative_prompt : Optional[str]
            Prompt specifying what not to generate.
        model_name : Optional[str]
            The model to use. If None, selects the best image generation model.
        image_width : int
            Desired width in pixels.
        image_height : int
            Desired height in pixels.
        preserve_aspect_ratio : bool
            Whether to preserve the aspect ratio.
        allow_resizing : bool
            Whether to allow resizing to match model constraints.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Image
            The generated PIL Image.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.ImageGeneration)

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the generate_image method of the api object:
        image_uri = api.generate_image(
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt,
            model_name=model_name,
            image_width=image_width,
            image_height=image_height,
            preserve_aspect_ratio=preserve_aspect_ratio,
            allow_resizing=allow_resizing,
        )

        # Update kwargs with other parameters:
        kwargs.update(
            {
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "model_name": model_name,
                "image_width": image_width,
                "image_height": image_height,
                "preserve_aspect_ratio": preserve_aspect_ratio,
                "allow_resizing": allow_resizing,
            }
        )

        # Call the callback manager if available:
        self.callback_manager.on_image_generation(positive_prompt, image_uri, **kwargs)

        # return the result:
        return image_uri

    def embed_texts(
        self,
        texts: Sequence[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Embed texts by delegating to the appropriate provider API.

        Parameters
        ----------
        texts : Sequence[str]
            The text snippets to embed.
        model_name : Optional[str]
            The model to use. If None, selects the best text embedding model.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input text.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.TextEmbeddings)

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the embed_texts method of the api object:
        embeddings = api.embed_texts(texts, model_name, dimensions)

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager if available:
        self.callback_manager.on_text_embedding(texts, embeddings, **kwargs)

        # return the result:
        return embeddings

    def embed_images(
        self,
        image_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Embed images by delegating to the appropriate provider API.

        Parameters
        ----------
        image_uris : List[str]
            List of image URIs or file paths.
        model_name : Optional[str]
            The model to use. If None, selects the best image embedding model.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input image.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.ImageEmbeddings)

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the embed_images method of the api object:
        embeddings = api.embed_images(
            image_uris=image_uris, model_name=model_name, dimensions=dimensions
        )

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager if available:
        self.callback_manager.on_image_embedding(image_uris, embeddings, **kwargs)

        # return the result:
        return embeddings

    def embed_audios(
        self,
        audio_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Embed audio files by delegating to the appropriate provider API.

        Parameters
        ----------
        audio_uris : List[str]
            List of audio URIs or file paths.
        model_name : Optional[str]
            The model to use. If None, selects the best audio embedding model.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input audio.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.AudioEmbeddings)

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the embed_audios method of the api object:
        embeddings = api.embed_audios(audio_uris, model_name, dimensions)

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager if available:
        self.callback_manager.on_audio_embedding(audio_uris, embeddings, **kwargs)

        # return the result:
        return embeddings

    def embed_videos(
        self,
        video_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Embed videos by delegating to the appropriate provider API.

        Parameters
        ----------
        video_uris : List[str]
            List of video URIs or file paths.
        model_name : Optional[str]
            The model to use. If None, selects the best video embedding model.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input video.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.VideoEmbeddings)

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the embed_videos method of the api object:
        embeddings = api.embed_videos(video_uris, model_name, dimensions)

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager if available:
        self.callback_manager.on_video_embedding(video_uris, embeddings, **kwargs)

        # return the result:
        return embeddings

    def embed_documents(
        self,
        document_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Embed documents by delegating to the appropriate provider API.

        Parameters
        ----------
        document_uris : List[str]
            List of document URIs or file paths.
        model_name : Optional[str]
            The model to use. If None, selects the best document embedding model.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input document.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.DocumentEmbeddings)

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the embed_documents method of the api object:
        embeddings = api.embed_documents(document_uris, model_name, dimensions)

        # Update kwargs with other parameters:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager if available:
        self.callback_manager.on_document_embedding(document_uris, embeddings, **kwargs)

        # return the result:
        return embeddings

    def describe_image(
        self,
        image_uri: str,
        system: str = "You are a helpful AI assistant that can describe/analyse images.",
        query: str = "Here is an image, please carefully describe it in detail.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> str:
        """Describe an image by delegating to the appropriate provider API.

        Parameters
        ----------
        image_uri : str
            Path, URL, or base64-encoded image to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the image.
        model_name : Optional[str]
            Model to use. If None, selects the best image-capable model.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure.

        Returns
        -------
        str
            Text description of the image.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(
                features=[ModelFeatures.TextGeneration, ModelFeatures.Image]
            )

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the describe_image method of the api object:
        desc = api.describe_image(
            image_uri,
            system,
            query,
            model_name,
            temperature,
            max_output_tokens,
            number_of_tries,
        )

        # Set kwargs with other parameters:
        kwargs = {
            "system": system,
            "query": query,
            "model_name": model_name,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "number_of_tries": number_of_tries,
        }

        # Call the callback manager if available:
        self.callback_manager.on_image_description(image_uri, desc, **kwargs)

        # return the result:
        return desc

    def describe_audio(
        self,
        audio_uri: str,
        system: str = "You are a helpful AI assistant that can describe/analyse audio.",
        query: str = "Here is an audio file, please carefully describe it in detail. If it is speach, please transcribe accurately.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> str:
        """Describe an audio file by delegating to the appropriate provider API.

        Parameters
        ----------
        audio_uri : str
            Path, URL, or base64-encoded audio to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the audio.
        model_name : Optional[str]
            Model to use. If None, selects the best audio-capable model.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure.

        Returns
        -------
        str
            Text description of the audio.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(
                features=[ModelFeatures.TextGeneration, ModelFeatures.Audio]
            )

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the describe_audio method of the api object:
        desc = api.describe_audio(
            audio_uri,
            system,
            query,
            model_name,
            temperature,
            max_output_tokens,
            number_of_tries,
        )

        # Set kwargs with other parameters:
        kwargs = {
            "system": system,
            "query": query,
            "model_name": model_name,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "number_of_tries": number_of_tries,
        }

        # Call the callback manager if available:
        self.callback_manager.on_audio_description(audio_uri, desc, **kwargs)

        # return the result:
        return desc

    def transcribe_audio(
        self, audio_uri: str, model_name: Optional[str] = None, **model_kwargs
    ) -> str:
        """Transcribe audio by delegating to the appropriate provider API.

        Parameters
        ----------
        audio_uri : str
            Path, URL, or base64-encoded audio to transcribe.
        model_name : Optional[str]
            The model to use. If None, selects the best transcription model.
        **model_kwargs
            Additional provider-specific arguments.

        Returns
        -------
        str
            The transcription of the audio.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(
                features=[ModelFeatures.AudioTranscription]
            )

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the transcribe_audio method of the api object:
        transcription = api.transcribe_audio(
            audio_uri=audio_uri, model_name=model_name, **model_kwargs
        )

        # Set kwargs with other parameters:
        kwargs = {"model_name": model_name, "model_kwargs": model_kwargs}

        # Call the callback manager if available:
        self.callback_manager.on_audio_transcription(transcription, audio_uri, **kwargs)

        # return the result:
        return transcription

    def describe_video(
        self,
        video_uri: str,
        system: str = "You are a helpful AI assistant that can describe/analyse videos.",
        query: str = "Here is a video file, please carefully describe it in detail.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> str:
        """Describe a video by delegating to the appropriate provider API.

        Parameters
        ----------
        video_uri : str
            Path, URL, or base64-encoded video to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the video.
        model_name : Optional[str]
            Model to use. If None, selects the best video-capable model.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure.

        Returns
        -------
        str
            Text description of the video.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(
                features=[ModelFeatures.TextGeneration, ModelFeatures.Video]
            )

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the describe_video method of the api object:
        desc = api.describe_video(
            video_uri,
            system,
            query,
            model_name,
            temperature,
            max_output_tokens,
            number_of_tries,
        )

        # Set kwargs with other parameters:
        kwargs = {
            "system": system,
            "query": query,
            "model_name": model_name,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "number_of_tries": number_of_tries,
        }

        # Call the callback manager if available:
        self.callback_manager.on_video_description(video_uri, desc, **kwargs)

        # return the result:
        return desc

    def describe_document(
        self,
        document_uri: str,
        system: str = "You are a helpful AI assistant that can describe/analyse documents.",
        query: str = "Here is a document file, please carefully describe it in detail.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> str:
        """Describe a document by delegating to the appropriate provider API.

        Parameters
        ----------
        document_uri : str
            Path, URL, or base64-encoded document to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the document.
        model_name : Optional[str]
            Model to use. If None, selects the best document-capable model.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure.

        Returns
        -------
        str
            Text description of the document.

        Raises
        ------
        ValueError
            If the resolved model is not found in any combined API.
        """
        # if no model name is provided we return the best model as defined in the method above:
        if model_name is None:
            model_name = self.get_best_model(
                features=[ModelFeatures.TextGeneration, ModelFeatures.Document]
            )

        # if the model name is not in the model_to_api dictionary we raise an error:
        if model_name not in self.model_to_api:
            raise ValueError(f"Model '{model_name}' not found in any API.")

        # we get the api object from the model_to_api dictionary:
        api = self.model_to_api[model_name]

        # we call the describe_video method of the api object:
        desc = api.describe_document(
            document_uri,
            system,
            query,
            model_name,
            temperature,
            max_output_tokens,
            number_of_tries,
        )

        # Set kwargs with other parameters:
        kwargs = {
            "system": system,
            "query": query,
            "model_name": model_name,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "number_of_tries": number_of_tries,
        }

        # Call the callback manager if available:
        self.callback_manager.on_document_description(document_uri, desc, **kwargs)

        # return the result:
        return desc
