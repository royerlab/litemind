from typing import Dict, List, Optional, Sequence, Type, Union

from arbol import aprint
from PIL.Image import Image
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.base_api import BaseApi
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.default_api import DefaultApi
from litemind.apis.exceptions import APIError
from litemind.apis.model_features import ModelFeatures
from litemind.media.media_base import MediaBase


class CombinedApi(DefaultApi):
    """
    A class that combines multiple APIs into a single API.
    It checks the availability and credentials of each API and adds them to the list.

    API Methods will be called on the first available API that supports the required features.

    """

    def __init__(
        self,
        apis: Optional[List[BaseApi]] = None,
        api_keys: Optional[Dict[str, str]] = None,
        callback_manager: Optional[CallbackManager] = None,
    ):
        """
        Create a new combined API. If no APIs are provided, the default APIs will be used: OpenAIApi, AnthropicApi, OllamaApi, GeminiApi.

        Parameters
        ----------
        apis: List[BaseApi]
            A list of APIs to combine.
        api_keys: Dict[str, str]
            A dictionary of API keys for each API.
        callback_manager: CallbackManager
            A callback manager to handle callbacks.
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
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        exclusion_filters: Optional[List[str]] = None,
    ) -> Optional[str]:

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
