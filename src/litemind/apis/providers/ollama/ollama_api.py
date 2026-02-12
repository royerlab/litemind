"""Ollama API provider implementation.

This module implements the ``OllamaApi`` class, which wraps the Ollama
local inference server to provide text generation, text embeddings, and
optional image understanding through a unified interface conforming to
``BaseApi``/``DefaultApi``.

Ollama models run locally and do not require API keys. The server must
be running (typically at ``http://localhost:11434``) before this API can
be used. Models are identified by their Ollama name (e.g.,
``llama3:latest``), and thinking-mode variants are created automatically
by appending ``-thinking`` to each model name.
"""

from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Type, Union

from arbol import aprint

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.api_callback_manager import ApiCallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.model_registry import get_default_model_registry
from litemind.apis.providers.ollama.utils.aggregate_chat_responses import (
    aggregate_chat_responses,
)
from litemind.apis.providers.ollama.utils.check_availability import (
    check_ollama_api_availability,
)
from litemind.apis.providers.ollama.utils.convert_messages import (
    convert_messages_for_ollama,
)
from litemind.apis.providers.ollama.utils.format_tools import format_tools_for_ollama
from litemind.apis.providers.ollama.utils.list_models import _get_ollama_models_list
from litemind.apis.providers.ollama.utils.process_response import (
    process_response_from_ollama,
)
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_image import Image
from litemind.media.types.media_text import Text


class OllamaApi(DefaultApi):
    """
    Ollama API client for interacting with the Ollama server that conforms to the BaseApi interface.

    Ollama models support text generation, some support image inputs and thinking,
    but do not support all features natively. Support for these features
    are provided by the Litemind API via our fallback mechanism.

    This class provides methods to check API availability, list models, get the best model,
    check model support for specific features, and generate text using the Ollama API.

    """

    def __init__(
        self,
        host: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        allow_media_conversions: bool = True,
        allow_media_conversions_with_models: bool = True,
        callback_manager: Optional[ApiCallbackManager] = None,
        **kwargs,
    ):
        """Initialize the Ollama API client.

        Parameters
        ----------
        host : Optional[str]
            The host URL for the Ollama server (default is
            ``http://localhost:11434``).
        headers : Optional[Dict[str, str]]
            Additional HTTP headers to pass to the Ollama client.
        allow_media_conversions : bool
            If True, the API will allow media conversions using the
            default media converter.
        allow_media_conversions_with_models : bool
            If True, the API will allow media conversions using models
            that support the required features, in addition to the
            default media converter. Requires ``allow_media_conversions``
            to be True.
        callback_manager : Optional[ApiCallbackManager]
            Optional callback manager for lifecycle event notifications.
        **kwargs
            Additional options passed to the Ollama ``Client(...)``.

        Raises
        ------
        APINotAvailableError
            If the Ollama server is not reachable or the client cannot
            be initialised.
        """

        super().__init__(
            allow_media_conversions=allow_media_conversions,
            allow_media_conversions_with_models=allow_media_conversions_with_models,
            callback_manager=callback_manager,
        )

        try:
            # Initialize the model registry for feature lookups:
            self.model_registry = get_default_model_registry()

            # Connect to Ollama server:
            from ollama import Client

            self.client = Client(host=host, headers=headers, **kwargs)

            # Save some fields:
            self.host = host
            self.headers = headers

            # Fetch raw model list:
            self._model_list = _get_ollama_models_list(self.client)

        except Exception as e:
            raise APINotAvailableError(f"Ollama server is not available: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:
        """Check if the Ollama server is available and responding.

        Since Ollama runs locally, no API key validation is performed.
        The ``api_key`` parameter is accepted for interface compatibility
        but is ignored.

        Parameters
        ----------
        api_key : Optional[str]
            Ignored. Ollama does not require API keys.

        Returns
        -------
        bool
            True if the Ollama server is reachable.
        """

        # Check if the Ollama API is available:
        result = check_ollama_api_availability(self.client)

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
        """List available Ollama models, optionally filtered by features.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required model features to filter by.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features that models must *not* have.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the models must support.

        Returns
        -------
        List[str]
            List of model name strings matching the criteria.

        Raises
        ------
        APIError
            If the model list cannot be retrieved from Ollama.
        """

        try:
            # Normalise the features:
            features = ModelFeatures.normalise(features)
            non_features = ModelFeatures.normalise(non_features)

            # Get raw Ollama model list:
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
            raise APIError("Error fetching model list from Ollama.")

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
        """Get the best available Ollama model for the required features.

        Models are sorted by size (largest first), so the first matching
        model is the most capable one available.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required model features.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features that the model must *not* have.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must support.
        exclusion_filters : Optional[Union[str, List[str]]]
            Substring patterns to exclude from model names.

        Returns
        -------
        Optional[str]
            The name of the best matching model, or None if no model matches.
        """

        # Normalise the features:
        features = ModelFeatures.normalise(features)
        non_features = ModelFeatures.normalise(non_features)

        # Get the list of models:
        model_list = self.list_models()

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
        """Check if a specific Ollama model supports the given features.

        Checks the model registry and the parent class fallback models
        for feature support.

        Parameters
        ----------
        features : Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            The features to check for.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must support.
        model_name : Optional[str]
            The model to check. If None, the best model for the given
            features is selected automatically.

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

        for feature in features:
            if not self.model_registry.supports_feature(
                self.__class__, model_name, feature
            ):
                # If the model does not support the feature, we return False:
                return False

        return True

    def _is_ollama_model(self, model_name: str) -> bool:
        """Check if a model name looks like an Ollama model.

        Ollama model names contain a colon separator (e.g.,
        ``llama3:latest``).

        Parameters
        ----------
        model_name : str
            The model name to check.

        Returns
        -------
        bool
            True if the name contains a colon.
        """
        return ":" in model_name

    def _has_thinking_support(self, model_name: str) -> bool:
        """Check if a model name indicates thinking-mode support.

        Thinking-mode variants are identified by the ``-thinking`` suffix
        (e.g., ``llama3:latest-thinking``).

        Parameters
        ----------
        model_name : str
            The model name to check.

        Returns
        -------
        bool
            True if the model name ends with ``-thinking``.
        """
        if model_name.endswith("-thinking"):
            return True
        return False

    @lru_cache
    def _has_tool_support(self, model_name: str) -> bool:
        """Check if an Ollama model supports tool/function calling.

        Inspects the model's Modelfile template for the ``.Tools``
        template directive. Results are cached for performance.

        Parameters
        ----------
        model_name : str
            The Ollama model name to check.

        Returns
        -------
        bool
            True if the model template contains tool-calling support.
        """
        if not self._is_ollama_model(model_name):
            # All Ollama models support text generation and contain ':'
            return False

        try:
            model_template = self.client.show(model_name).template
            return ".Tools" in model_template
        except Exception:
            return False

    @lru_cache
    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum context length for the given Ollama model.

        Queries the model's metadata for the ``context_length`` parameter.
        Results are cached for performance. Falls back to 2,500 if the
        information cannot be retrieved.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query. If None, the best text-generation model
            is used.

        Returns
        -------
        int
            Maximum number of input tokens (context window size).
        """
        from ollama import ResponseError

        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.TextGeneration)

        try:
            model_info = self.client.show(model_name).modelinfo

            # search key that contains 'context_length'
            for key in model_info.keys():
                if "context_length" in key:
                    return model_info[key]
        except ResponseError as e:
            aprint(f"Model {model_name} not found in Ollama: {e.error}")

        # return default value if nothing else works:
        return 2500

    @lru_cache
    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of output tokens for the given Ollama model.

        Queries the model's metadata for the ``max_tokens`` parameter.
        Results are cached for performance. Falls back to 4,096 if the
        information cannot be retrieved.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query. If None, the best text-generation model
            is used.

        Returns
        -------
        int
            Maximum number of output tokens the model can generate.
        """
        from ollama import ResponseError

        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.TextGeneration)

        try:
            model_info = self.client.show(model_name).modelinfo

            # search key that contains 'max_tokens'
            for key in model_info.keys():
                if "max_tokens" in key:
                    return model_info[key]
        except ResponseError as e:
            aprint(f"Model {model_name} not found in Ollama: {e.error}")

        # return default value if nothing else works:
        return 4096

    def generate_text(
        self,
        messages: List[Message],
        model_name: Optional[str] = None,
        temperature: Optional[float] = 0.0,
        max_num_output_tokens: Optional[int] = None,
        toolset: Optional[ToolSet] = None,
        use_tools: bool = True,
        response_format: Optional["BaseModel"] = None,
        **kwargs,
    ) -> List[Message]:
        """Generate text using the Ollama chat API with streaming.

        Supports multi-turn conversations, tool calling with automatic
        execution loops, thinking-mode models, and optional structured
        output via a Pydantic ``response_format``.

        Parameters
        ----------
        messages : List[Message]
            The input messages for the conversation.
        model_name : Optional[str]
            The Ollama model to use (e.g., ``llama3:latest``). If None,
            the best available text-generation model is selected.
        temperature : Optional[float]
            Sampling temperature for text generation.
        max_num_output_tokens : Optional[int]
            Maximum number of tokens to generate.
        toolset : Optional[ToolSet]
            Tools to make available to the model for function calling.
        use_tools : bool
            Whether to execute tool calls returned by the model.
        response_format : Optional[BaseModel]
            Pydantic model for structured JSON output.
        **kwargs
            Additional keyword arguments passed to the Ollama chat API.

        Returns
        -------
        List[Message]
            The newly generated response messages (may include tool-use
            messages if tools were invoked).

        Raises
        ------
        APIError
            If the Ollama API returns an error during generation.
        """

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

        # Ollama imports:
        from ollama import ResponseError

        # Set default model if not provided
        if model_name is None:
            model_name = self._get_best_model_for_text_generation(
                messages, toolset if use_tools else None, response_format
            )

        # Initialize the allowed media types:
        allowed_media_types: Set[Type[MediaBase]] = {Text}

        # Check if the model supports images:
        # TODO: add way to force image support so that the scanner can figure out if models really support images
        if self.has_model_support_for(
            model_name=model_name, features=ModelFeatures.Image
        ):
            allowed_media_types.add(Image)

        # Preprocess the messages:
        preprocessed_messages = self._preprocess_messages(
            messages=messages,
            allowed_media_types=self._get_allowed_media_types_for_text_generation(
                model_name=model_name
            ),
        )

        # If this is a thinking model then we need to add a system message about thinking:
        if self._has_thinking_support(model_name):

            # Remove the -thinking postfix from the model name:
            if model_name.endswith("-thinking"):
                model_name = model_name[:-9]

            # Find the first system message in  preprocessed_messages:
            system_message = next(
                (m for m in preprocessed_messages if m.role == "system"), None
            )
            # Append the instruction to the system message:
            if not system_message:
                # Create a new empty system message and add it at the beginning:
                system_message = Message(role="system", text="")
                preprocessed_messages.insert(0, system_message)

            # Find the first text block in the system message:
            for block in system_message.blocks:
                if block.has_type(Text):
                    text: str = block.media.text

                    # Append the instruction to the text block:
                    text += "\n"
                    text += "Think carefully step-by-step before responding: restate the input, analyze it, consider options, make a plan, and proceed methodically to your conclusion. \n"
                    text += "All reasoning (thinking) which precedes the final answer must be enclosed within thinking tags: <thinking> reasoning goes here... </thinking> final answer here...\n\n"
                    block.media.text = text
                    break

        # Get max num of output tokens for model if not provided:
        if max_num_output_tokens is None:
            max_num_output_tokens = self.max_num_output_tokens(model_name)

        # Convert toolset (if any) to Ollama's tools schema
        ollama_tools = format_tools_for_ollama(toolset) if toolset else None

        # List of new messages part of the response:
        new_messages = []

        try:
            # Loop until we get a response that doesn't require tool use:
            while True:

                # Convert user messages into Ollama's format
                ollama_formatted_messages = convert_messages_for_ollama(
                    preprocessed_messages, response_format=response_format
                )

                # Call the Ollama API in streaming mode:
                chunks = self.client.chat(
                    model=model_name,
                    messages=ollama_formatted_messages,
                    tools=ollama_tools,
                    options={
                        "temperature": temperature,
                        "num_predict": max_num_output_tokens,
                    },
                    stream=True,
                    **kwargs,
                )

                # Aggregate the streamed response:
                ollama_response = aggregate_chat_responses(
                    chunks, callback=self.callback_manager.on_text_streaming
                )

                # Process the response
                response = process_response_from_ollama(
                    ollama_response, toolset, response_format
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

        except ResponseError as e:
            raise APIError(
                f"Ollama generate text error: {e.error} (status code: {e.status_code})"
            )

        return new_messages

    def embed_texts(
        self,
        texts: Sequence[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Generate text embeddings using an Ollama embedding model.

        If the model's native embedding dimension differs from the
        requested ``dimensions``, the vectors are resized automatically
        using deterministic random projection.

        Parameters
        ----------
        texts : Sequence[str]
            The texts to embed.
        model_name : Optional[str]
            The embedding model to use. If None, the best available
            embedding model is selected automatically.
        dimensions : int
            The desired dimensionality of the embedding vectors.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            A sequence of embedding vectors, one per input text.
        """

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model(features=ModelFeatures.TextEmbeddings)

        # If the model belongs to the superclass, we call the superclass method:
        if model_name in super().list_models():
            return super().embed_texts(
                texts=texts, model_name=model_name, dimensions=dimensions, **kwargs
            )

        from ollama import EmbedResponse

        embed_response: EmbedResponse = self.client.embed(model=model_name, input=texts)

        # Extract embeddings:
        embeddings = embed_response.embeddings

        # Check that the embeddings are of the correct dimension:
        if len(embeddings[0]) != dimensions:
            # use function _reduce_embdedding_dimension to reduce or increase the dimension
            resized_embeddings = self._reduce_embeddings_dimension(
                embeddings, dimensions
            )
            embeddings = resized_embeddings

        # Update the kwargs with the model name and dimensions:
        kwargs.update({"model_name": model_name, "dimensions": dimensions})

        # Call the callback manager:
        self.callback_manager.on_text_embedding(
            texts=texts, embeddings=embeddings, **kwargs
        )

        return embeddings
