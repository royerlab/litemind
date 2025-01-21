from typing import List, Dict, Optional, Sequence, Union

from arbol import aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi, ModelFeatures
from litemind.apis.exceptions import APIError, APINotAvailableError
from litemind.apis.ollama.utils.messages import convert_messages_for_ollama
from litemind.apis.ollama.utils.process_response import _process_response
from litemind.apis.ollama.utils.tools import _format_tools_for_ollama
from litemind.apis.utils.document_processing import is_pymupdf_available
from litemind.apis.utils.whisper_transcribe_audio import \
    is_local_whisper_available


class OllamaApi(BaseApi):
    def __init__(self,
                 host: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None,
                 **kwargs):
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
        try:
            from ollama import Client
            self.client = Client(host=host,
                                 headers=headers,
                                 **kwargs)

            self.host = host
            self.headers = headers

            self._model_list_names = [model.model for model in
                                      self.client.list().models]
            self._model_list_sizes = [model.size for model in
                                      self.client.list().models]
        except Exception as e:
            # Print stack trace:
            import traceback
            traceback.print_exc()
            raise APINotAvailableError(
                f"Error initializing Ollama client: {e}")

    def model_list(self, features: Optional[Sequence[ModelFeatures]] = None) -> \
            List[str]:

        try:
            # Sort by decreasing ollama model size:
            model_list = [x for _, x in sorted(zip(self._model_list_sizes,
                                                   self._model_list_names),
                                               reverse=True)]

            # Filter the models based on the features:
            if features:
                model_list = self._filter_models(model_list, features=features)

            return model_list
        except Exception:
            raise APIError("Error fetching model list from Ollama.")

    def get_best_model(self, features: Optional[Union[
        str, List[str], ModelFeatures, Sequence[ModelFeatures]]] = None) -> \
            Optional[str]:

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Get the list of models:
        model_list = self.model_list()

        # Filter the models based on the requirements:
        model_list = self._filter_models(model_list,
                                         features=features)

        # If we have any models left, return the first one
        if model_list:
            return model_list[0]
        else:
            return None

    def check_availability_and_credentials(self, api_key: Optional[
        str] = None) -> bool:

        try:
            self.client.list()
            return True
        except Exception:
            # If we get an error, we assume it's because Ollama is not running:
            return False

    def has_model_support_for(self,
                              features: Union[
                                  str, List[str], ModelFeatures, Sequence[
                                      ModelFeatures]],
                              model_name: Optional[str] = None) -> bool:

        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model()

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Check that the model has all the required features:
        for feature in features:

            if feature == ModelFeatures.TextGeneration:
                pass

            elif feature == ModelFeatures.ImageGeneration:
                return False

            elif feature == ModelFeatures.TextEmbeddings:
                pass

            elif feature == ModelFeatures.ImageEmbeddings:
                if not self.has_model_support_for(
                        [ModelFeatures.Image, ModelFeatures.TextEmbeddings],
                        model_name):
                    return False

            elif feature == ModelFeatures.AudioEmbeddings:
                if not self.has_model_support_for(
                        [ModelFeatures.Audio, ModelFeatures.TextEmbeddings],
                        model_name):
                    return False

            elif feature == ModelFeatures.VideoEmbeddings:
                if not self.has_model_support_for(
                        [ModelFeatures.Video, ModelFeatures.TextEmbeddings],
                        model_name):
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

            elif feature == ModelFeatures.Tools:
                if not self._has_tool_support(model_name):
                    return False

            else:
                if not super().has_model_support_for(feature, model_name):
                    return False

        return True

    def _has_image_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.get_best_model()

        if 'vision' in model_name:
            return True

        try:
            model_details_families = self.client.show(
                model_name).details.families
            return model_details_families and (
                    'clip' in model_details_families or 'mllama' in model_details_families)
        except:
            return False

    def _has_audio_support(self, model_name: Optional[str] = None) -> bool:

        # No Ollama models currently support Audio, but we use local whisper as a fallback:
        return is_local_whisper_available()

    def _has_tool_support(self, model_name: Optional[str] = None) -> bool:

        if model_name is None:
            model_name = self.get_best_model()

        try:
            model_template = self.client.show(model_name).template
            return '$.Tools' in model_template
        except:
            return False

    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:

        if model_name is None:
            model_name = self.get_best_model()

        model_info = self.client.show(model_name).modelinfo

        # search key that contains 'context_length'
        for key in model_info.keys():
            if 'context_length' in key:
                return model_info[key]

        # return default value if nothing else works:
        return 2500

    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:

        if model_name is None:
            model_name = self.get_best_model()

        model_info = self.client.show(model_name).modelinfo

        # search key hat contains 'max_tokens'
        for key in model_info.keys():
            if 'max_tokens' in key:
                return model_info[key]

        # return default value if nothing else works:
        return 4096

    def generate_text_completion(self,
                                 messages: List[Message],
                                 model_name: Optional[str] = None,
                                 temperature: Optional[float] = 0.0,
                                 max_output_tokens: Optional[int] = None,
                                 toolset: Optional[ToolSet] = None,
                                 **kwargs) -> Message:

        # Set default model if not provided
        if model_name is None:
            model_name = self.get_best_model()

        # Get max num of output tokens for model if not provided:
        if max_output_tokens is None:
            max_output_tokens = self.max_num_output_tokens(model_name)

        # We will use _messages to process the messages:
        preprocessed_messages = messages

        # Convert video URIs to images and audio because Ollama does not natively support videos:
        preprocessed_messages = self._convert_videos_to_images_and_audio(
            preprocessed_messages)

        # If model does not support audio but audio transcription is available, then we use it to transcribe audio:
        if self.has_model_support_for(model_name=model_name,
                                      features=ModelFeatures.AudioTranscription):
            preprocessed_messages = self._transcribe_audio_in_messages(
                preprocessed_messages)

        # Convert documents to markdown and images:
        if is_pymupdf_available():
            preprocessed_messages = self._convert_documents_to_markdown_in_messages(
                preprocessed_messages)

        # Convert user messages into Ollama's format
        ollama_messages = convert_messages_for_ollama(preprocessed_messages)

        # Convert toolset (if any) to Ollama's tools schema
        ollama_tools = _format_tools_for_ollama(toolset) if toolset else None

        # Make the request to Ollama
        from ollama import ResponseError
        try:
            aprint(f"Sending request to Ollama with model: {model_name}")
            response = self.client.chat(
                model=model_name,
                messages=ollama_messages,
                tools=ollama_tools,
                options={"temperature": temperature,
                         "num_predict": max_output_tokens},
                **kwargs
            )

            # 4) Process the response
            response_message = _process_response(response, toolset)
            messages.append(response_message)
            return response_message

        except ResponseError as e:
            raise APIError(
                f"Error during completion: {e.error} (status code: {e.status_code})")

    def embed_texts(self,
                    texts: List[str],
                    model_name: Optional[str] = None,
                    dimensions: int = 512,
                    **kwargs) -> Sequence[Sequence[float]]:

        if model_name is None:
            model_name = self.get_best_model(require_embeddings=True)

        from ollama import EmbedResponse
        embed_response: EmbedResponse = self.client.embed(model=model_name,
                                                          input=texts)

        # Extract embeddings:
        embeddings = embed_response.embeddings

        # Check that the embeddings are of the correct dimension:
        if len(embeddings[0]) != dimensions:
            # use function _reduce_embdedding_dimension to reduce or increase the dimension
            resized_embeddings = self._reduce_embdeddings_dimension(embeddings,
                                                                    dimensions)
            return resized_embeddings
        else:
            return embeddings
