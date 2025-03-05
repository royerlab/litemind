from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Union

from arbol import aprint

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import ModelFeatures
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.providers.ollama.utils.aggregate_chat_responses import (
    aggregate_chat_responses,
)
from litemind.apis.providers.ollama.utils.convert_messages import (
    convert_messages_for_ollama,
)
from litemind.apis.providers.ollama.utils.format_tools import format_tools_for_ollama
from litemind.apis.providers.ollama.utils.list_models import _get_ollama_models_list
from litemind.apis.providers.ollama.utils.process_response import (
    process_response_from_ollama,
)


class OllamaApi(DefaultApi):
    def __init__(
        self,
        host: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        callback_manager: Optional[CallbackManager] = None,
        **kwargs,
    ):
        """
        Initialize the Ollama API client.

        Parameters
        ----------
        host: Optional[str]
            The host for the Ollama client (default is `http://localhost:11434`).
        headers:
            Additional headers to pass to the client.
        kwargs:
            Additional options passed to the `Client(...)`.
        """

        super().__init__(callback_manager=callback_manager)

        try:
            from ollama import Client

            # Connect to Ollama server:
            self.client = Client(host=host, headers=headers, **kwargs)

            # Save some fields:
            self.host = host
            self.headers = headers

            # Fetch raw model list:
            self._model_list = _get_ollama_models_list(self.client)

        except Exception as e:
            # Print stack trace:
            import traceback

            traceback.print_exc()
            raise APINotAvailableError(f"Error initializing Ollama client: {e}")

    def check_availability_and_credentials(self, api_key: Optional[str] = None) -> bool:

        try:
            self.client.list()
            self.callback_manager.on_availability_check(True)
            return True
        except Exception:
            # If we get an error, we assume it's because Ollama is not running:
            self.callback_manager.on_availability_check(False)
            return False

    def list_models(
        self, features: Optional[Sequence[ModelFeatures]] = None
    ) -> List[str]:

        try:

            # Get raw Ollama model list:
            model_list = list(self._model_list)

            # Add the models from the super class:
            model_list += super().list_models()

            # Filter the models based on the features:
            if features:
                model_list = self._filter_models(model_list, features=features)

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
        exclusion_filters: Optional[Union[str, List[str]]] = None,
    ) -> Optional[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the list of models:
        model_list = self.list_models()

        # Filter the models based on the requirements:
        model_list = self._filter_models(
            model_list, features=features, exclusion_filters=exclusion_filters
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
                if not self._has_text_generation_support(model_name):
                    return False

            elif feature == ModelFeatures.TextEmbeddings:
                if not self._is_ollama_model(model_name):
                    # All Ollama models support text generation and contain ':'
                    return False

            elif feature == ModelFeatures.ImageEmbeddings:
                if not self.has_model_support_for(
                    [ModelFeatures.Image, ModelFeatures.TextEmbeddings], model_name
                ):
                    return False

            elif feature == ModelFeatures.AudioEmbeddings:
                if not self.has_model_support_for(
                    [ModelFeatures.Audio, ModelFeatures.TextEmbeddings], model_name
                ):
                    return False

            elif feature == ModelFeatures.VideoEmbeddings:
                if not self.has_model_support_for(
                    [ModelFeatures.Video, ModelFeatures.TextEmbeddings], model_name
                ):
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

    def _is_ollama_model(self, model_name: str) -> bool:
        return ":" in model_name

    def _has_text_generation_support(self, model_name: str) -> bool:
        if not self._is_ollama_model(model_name):
            # All Ollama models support text generation and contain ':'
            return False
        return True

    @lru_cache
    def _has_image_support(self, model_name: str) -> bool:

        if not self._is_ollama_model(model_name):
            # All Ollama models support text generation and contain ':'
            return False

        if "vision" in model_name:
            return True

        try:
            model_details_families = self.client.show(model_name).details.families
            return model_details_families and (
                "clip" in model_details_families or "mllama" in model_details_families
            )
        except:
            return False

    def _has_audio_support(self, model_name: Optional[str] = None) -> bool:

        # No Ollama models currently support Audio:
        return False

    def _has_document_support(self, model_name: str) -> bool:
        return self.has_model_support_for(
            [ModelFeatures.Image, ModelFeatures.TextGeneration], model_name
        ) and self.has_model_support_for(ModelFeatures.VideoConversion)

    @lru_cache
    def _has_tool_support(self, model_name: str) -> bool:
        if not self._is_ollama_model(model_name):
            # All Ollama models support text generation and contain ':'
            return False

        try:
            model_template = self.client.show(model_name).template
            return "$.Tools" in model_template
        except:
            return False

    def _has_structured_output_support(self, model_name: str) -> bool:
        if not self._is_ollama_model(model_name):
            # All Ollama models support text generation and contain ':'
            return False

        return True

    @lru_cache
    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
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
            # Print stack trace:
            import traceback

            traceback.print_exc()

        # return default value if nothing else works:
        return 2500

    @lru_cache
    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        from ollama import ResponseError

        if model_name is None:
            model_name = self.get_best_model(ModelFeatures.TextGeneration)

        try:
            model_info = self.client.show(model_name).modelinfo

            # search key hat contains 'max_tokens'
            for key in model_info.keys():
                if "max_tokens" in key:
                    return model_info[key]
        except ResponseError as e:
            aprint(f"Model {model_name} not found in Ollama: {e.error}")
            # Print stack trace:
            import traceback

            traceback.print_exc()

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

        # Ollama imports:
        from ollama import ResponseError

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

        # Preprocess the messages:
        preprocessed_messages = self._preprocess_messages(messages=messages)

        # Get max num of output tokens for model if not provided:
        if max_num_output_tokens is None:
            max_num_output_tokens = self.max_num_output_tokens(model_name)

        # Convert toolset (if any) to Ollama's tools schema
        ollama_tools = format_tools_for_ollama(toolset) if toolset else None

        # Ollama specific: normalise response format to JSON schema string:
        response_format_schema = (
            response_format.model_json_schema() if response_format else response_format
        )

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
                if not response.has(BlockType.Tool):
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
                response=ollama_response, messages=messages, **kwargs
            )

        except ResponseError as e:
            raise APIError(
                f"Ollama generate text error: {e.error} (status code: {e.status_code})"
            )

        return new_messages

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
