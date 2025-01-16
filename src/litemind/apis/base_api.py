from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from PIL.Image import Image
from arbol import asection, aprint

from litemind.agent.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.model_features import ModelFeatures
from litemind.apis.utils.random_projector import DeterministicRandomProjector
from litemind.apis.utils.whisper_transcribe_audio import \
    is_local_whisper_available, transcribe_audio_with_local_whisper


class BaseApi(ABC):

    @abstractmethod
    def check_availability_and_credentials(self,
                                           api_key: Optional[str] = None) -> \
    Optional[bool]:
        """
        Check if this API is available and whether credentials are valid.
        If no API key is provided, checks if the API key available in the environment is valid.
        If no API key is required, returns None.

        Parameters
        ----------
        api_key: Optional[str]
            If needed, the API key to check.

        Returns
        -------
        bool
            True if the API key is valid, False otherwise, None if no credentials are required.

        """
        pass

    @abstractmethod
    def model_list(self, features: Optional[Sequence[ModelFeatures]] = None) -> \
    List[str]:
        """
        Get the list of models available that satisfy a given set of features.

        Returns
        -------
        List[str]
            The list of models available

        """
        pass

    @abstractmethod
    def get_best_model(self,
                       features: Optional[Union[
                           str, List[str], ModelFeatures, Sequence[
                               ModelFeatures]]] = None,
                       ) -> Optional[str]:
        """
        Get the name of the best possible model that satisfies a set of capability requirements.

        Parameters
        ----------
        features: Sequence[ModelFeatures], or ModelFeatures, or str, or List[str]

        Returns
        -------
        str
            The default model to use.
            None if no models satisfy the requirements

        """
        pass

    def _filter_models(self,
                       model_list: List[str],
                       features: Union[str, List[str], ModelFeatures, Sequence[
                           ModelFeatures]]) -> List[str]:
        """
        Filter the list of models based on the given features.
        Parameters
        ----------
        model_list: List[str]
            List of models.
        features: Sequence[ModelFeatures], or ModelFeatures, or str, or List[str]
            List of features to filter on.

        Returns
        -------
        List[str]
            Filtered list of models.

        """
        filtered_model_list = []

        if not features:
            # Nothing to filter:
            return model_list

        for model in model_list:
            if self.has_model_support_for(model_name=model, features=features):
                filtered_model_list.append(model)

        return filtered_model_list

    @abstractmethod
    def has_model_support_for(self, features: Union[
        str, List[str], ModelFeatures, Sequence[ModelFeatures]],
                              model_name: Optional[str] = None) -> bool:
        """
        Check if the given model supports the given features.
        If no model is provided the default, and ideally, best model, is used.

        Parameters
        ----------
        features: Sequence[ModelFeatures] or ModelFeatures, or str, or List[str]
            The feature(s) to check for.
        model_name: Optional[str]
            The name of the model to use.

        Returns
        -------
        bool
            True if the model has text generation support, False otherwise.

        """
        # Get the best model if not provided:
        if model_name is None:
            model_name = self.get_best_model()

        # Normalise the features:
        features = ModelFeatures.normalise(features)

        # Check that the model has all the required features:
        for feature in features:

            if feature == ModelFeatures.AudioTranscription:

                # We provide a default implementation for audio transcription using local whisper is available:
                if not is_local_whisper_available():
                    return False
            else:
                return False

        return True

    def get_model_features(self, model_name: str) -> List[ModelFeatures]:
        """
        Get the features of the given model.

        Parameters
        ----------
        model_name: str
            The name of the model to use.

        Returns
        -------
        List[ModelFeatures]
            The features of the model.

        """

        # List to store model features:
        model_features: List[ModelFeatures] = []

        # Get all the features:
        for feature in ModelFeatures:
            if self.has_model_support_for(model_name=model_name,
                                          features=feature):
                model_features.append(feature)

        return model_features

    @abstractmethod
    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Get the maximum number of input tokens for the given model.

        Parameters
        ----------
        model_name: Optional[str]
            The name of the model to use.

        Returns
        -------
        int
            The maximum number of input tokens for the model.

        """
        pass

    @abstractmethod
    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """
        Get the maximum number of output tokens for the given model.
        Parameters
        ----------
        model_name: Optional[str]
            The name of the model to use.

        Returns
        -------
        int
            The maximum number of output tokens for the model.

        """
        pass

    @abstractmethod
    def generate_text_completion(self,
                                 messages: List[Message],
                                 model_name: Optional[str] = None,
                                 temperature: float = 0.0,
                                 max_output_tokens: Optional[int] = None,
                                 toolset: Optional[ToolSet] = None,
                                 **kwargs) -> Message:
        """
        Generate a text completion using the given model for a given list of messages and parameters.

        Parameters
        ----------
        model_name: str
            The name of the model to use.
        messages: List[Message]
            The list of messages to send to the model.
        temperature: float
            The temperature to use.
        max_output_tokens: Optional[int]
            The maximum number of tokens to use.
        toolset: Optional[ToolSet]
            The toolset to use.
        kwargs: dict
            Additional arguments to pass to the completion function, this is API specific!

        Returns
        -------
        Message
            The response from the model.

        """
        pass

    def transcribe_audio(self, audio_uri: str, model_name: Optional[str] = None,
                         **kwargs) -> str:
        """
        Transcribe an audio file using the model.

        Parameters
        ----------
        audio_uri: str
            Can be: a path to an audio file, a URL to an audio file, or a base64 encoded audio.
        model_name: Optional[str]
            The name of the model to use.
        kwargs: dict
            Additional arguments to pass to the transcription function.

        Returns
        -------
        str
            The transcription of the audio.

        """

        transcription = transcribe_audio_with_local_whisper(audio_uri=audio_uri,
                                                            **kwargs)

        return transcription

    def _transcribe_audio_in_messages(self,
                                      messages: List[Message]) -> List[Message]:

        # Iterate over each message in the list:
        for message in messages:

            # If the message has audio_uris:
            if message.audio_uris:

                # Iterate over each audio URI in the message:
                for audio_uri in message.audio_uris:
                    try:
                        # We use a local instance of whisper instead:
                        transcription = self.transcribe_audio(audio_uri)

                        # Extract filename from URI:
                        original_filename = audio_uri.split("/")[-1]

                        # Add markdown quotes ''' around the transcribed text, and
                        # add prefix: "Transcription: " to the transcribed text:
                        transcription = f"\nTranscription of audio file '{original_filename}': \n'''\n{transcription}\n'''\n"

                        # Add the transcribed text to the message
                        message.append_text(transcription)
                    except Exception as e:
                        raise ValueError(
                            f"Could not transcribe audio from: '{audio_uri}', error: {e}")

                # If the audio was transcribed, remove the audio_uris from the message:
                message.audio_uris = []

        return messages

    def generate_image(self,
                       model_name: str,
                       positive_prompt: str,
                       negative_prompt: Optional[str] = None,
                       image_width: int = 512,
                       image_height: int = 512,
                       preserve_aspect_ratio: bool = True,
                       allow_resizing: bool = True,
                       **kwargs
                       ) -> Image:
        """
        Generate an image using the default or given model.

        Parameters
        ----------
        model_name: str
            The name of the model to use.
        positive_prompt: str
            The prompt to use for image generation.
        negative_prompt: str
            Prompt to specifying what not to generate
        image_width: int
            The width of the image.
        image_height: int
            The height of the image.
        preserve_aspect_ratio: bool
            Whether to preserve the aspect ratio of the image.
        allow_resizing: bool
            Whether to allow resizing of the image.
        kwargs: dict
            Additional arguments to pass to the image generation function.

        Returns
        -------

        Image
            The generated image.

        """
        raise NotImplementedError(
            "Image generation is not supported by this API.")

    def embed_texts(self,
                    texts: List[str],
                    model_name: Optional[str] = None,
                    dimensions: int = 512,
                    **kwargs) -> Sequence[Sequence[float]]:
        """
        Embed text using the model.

        Parameters
        ----------
        texts: List[str]
            The text snippets to embed.
        model_name: Optional[str]
            The name of the model to use.
        dimensions: int
            The number of dimensions for the embedding.
        kwargs: dict
            Additional arguments to pass to the embedding function.

        Returns
        -------
        Sequence[Sequence[float]]
            The embedding of the text.

        """
        raise NotImplementedError(
            "Text embedding is not supported by this API.")

    def _reduce_embdeddings_dimension(self,
                                      embeddings: Sequence[Sequence[float]],
                                      reduced_dim: int) -> Sequence[float]:
        drp = DeterministicRandomProjector(original_dim=len(embeddings[0]),
                                           reduced_dim=reduced_dim)
        return drp.transform(embeddings)

    def embed_images(self,
                     image_uris: str,
                     model_name: Optional[str] = None,
                     dimensions: int = 512,
                     **kwargs) -> Sequence[Sequence[float]]:

        # If no model is passed get a default model with image support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.Image])

        # Model must support images:
        if not model_name or not self.has_model_support_for(
                model_name=model_name, features=[ModelFeatures.Image]):
            raise ValueError(f"Model '{model_name}' does not support images.")

        # List to store image descriptions:
        image_descriptions = []

        # Iterate over the image_uris:
        for image_uri in image_uris:
            # Describe image and then embed the description of the image:
            description = self.describe_image(image_uri=image_uri,
                                              model_name=model_name,
                                              **kwargs)

            # Append the description to the list:
            image_descriptions.append(description)

        # Embed the image descriptions:
        image_embeddings = self.embed_texts(text=image_descriptions,
                                            model_name=model_name,
                                            dimensions=dimensions)
        # Return the image embeddings:
        return image_embeddings

    def embed_audios(self,
                     audio_uris: List[str],
                     model_name: Optional[str] = None,
                     dimensions: int = 512,
                     **kwargs) -> Sequence[Sequence[float]]:
        """
        Embed audios using the model.

        Parameters
        ----------
        audio_uris: List[str]
            List of Audio URIs.
        model_name: Optional[str]
            The name of the model to use.
        dimensions: int
            The number of dimensions for the embedding.
        kwargs: dict
            Additional arguments to pass to the embedding function.

        Returns
        -------
        Sequence[Sequence[float]]
            The embeddings of the audios.

        """
        # Implement similarly to embed_images:

        # If no model is passed get a default model with audio support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.Audio])

        # Model must support audios:
        if not model_name or not self.has_model_support_for(
                model_name=model_name, features=[ModelFeatures.Audio]):
            raise ValueError(f"Model '{model_name}' does not support audios.")

        # List to store audio descriptions:
        audio_descriptions = []

        # Iterate over the audio_uris:
        for audio_uri in audio_uris:
            # Describe video and then embed the description of the video:
            description = self.describe_audio(audio_uri=audio_uri,
                                              model_name=model_name,
                                              **kwargs)

            # Append the description to the list:
            audio_descriptions.append(description)

        # Embed the audio descriptions:
        audio_embeddings = self.embed_texts(text=audio_descriptions,
                                            model_name=model_name,
                                            dimensions=dimensions)

        # Return the audio embeddings:
        return audio_embeddings

    def embed_videos(self,
                     video_uris: List[str],
                     model_name: Optional[str] = None,
                     dimensions: int = 512,
                     **kwargs) -> Sequence[Sequence[float]]:
        """
        Embed videos using the model.

        Parameters
        ----------
        video_uris: List[str]
            List of video URIs.
        model_name: Optional[str]
            The name of the model to use.
        dimensions: int
            The number of dimensions for the embedding.
        kwargs: dict
            Additional arguments to pass to the embedding function.

        Returns
        -------
        Sequence[Sequence[float]]
            The embeddings of the videos.

        """
        # Implement similarly to embed_images:

        # If no model is passed get a default model with video support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.Video])

        # Model must support videos:
        if not model_name or not self.has_model_support_for(
                model_name=model_name, features=[ModelFeatures.Video]):
            raise ValueError(f"Model '{model_name}' does not support videos.")

        # List to store video descriptions:
        video_descriptions = []

        # Iterate over the video_uris:
        for video_uri in video_uris:
            # Describe video and then embed the description of the video:
            description = self.describe_video(video_uri=video_uri,
                                              model_name=model_name,
                                              **kwargs)

            # Append the description to the list:
            video_descriptions.append(description)

        # Embed the video descriptions:
        video_embeddings = self.embed_texts(text=video_descriptions,
                                            model_name=model_name,
                                            dimensions=dimensions)

        # Return the video embeddings:
        return video_embeddings

    def describe_image(self,
                       image_uri: str,
                       system: str = 'You are a helpful AI assistant that can describe/analyse images.',
                       query: str = 'Here is an image, please carefully describe it in detail.',
                       model_name: Optional[str] = None,
                       temperature: float = 0,
                       max_output_tokens: Optional[int] = None,
                       number_of_tries: int = 4,
                       ) -> str:
        """
        Describe an image using the model.

        Parameters
        ----------
        image_uri: str
            Can be: a path to an image file, a URL to an image, or a base64 encoded image.
        system:
            System message to use
        query: str
            Query to use with image
        model_name: Optional[str]
            Model to use
        temperature: float
            Temperature to use
        max_output_tokens: Optional[int]
            Maximum number of tokens to use
        number_of_tries: int
            Number of times to try to send the request to the model

        Returns
        -------
        str
            Description of the image

        """

        with (asection(
                f"Asking model {model_name} to descrbe a given image: '{image_uri}':")):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If no model is passed get a default model with image support:
            if model_name is None:
                model_name = self.get_best_model(require_images=True)

            # If the model does not support vision, return an error:
            if not self.has_model_support_for(model_name=model_name,
                                              features=ModelFeatures.Image):
                return f"Model '{model_name}' does not support images."

            # if no max_output_tokens is passed, get the default value:
            if max_output_tokens is None:
                max_output_tokens = self.max_num_output_tokens(model_name)
            else:
                # Limit the number of tokens to the maximum allowed by the model:
                max_output_tokens = min(max_output_tokens,
                                        self.max_num_output_tokens(model_name))

            try:

                image_model_name = self.get_best_model(ModelFeatures.Image)

                # Retry in case of model refusing to answer:
                for tries in range(number_of_tries):

                    messages = []

                    # System message:
                    system_message = Message(role='system')
                    system_message.append_text(system)
                    messages.append(system_message)

                    # User message:
                    user_message = Message(role='user')
                    user_message.append_text(query)
                    user_message.append_image_uri(image_uri)

                    messages.append(user_message)

                    # Run agent:
                    response = self.generate_text_completion(messages=messages,
                                                             model_name=image_model_name,
                                                             temperature=temperature,
                                                             max_output_tokens=max_output_tokens)

                    # Normalise response:
                    response = str(response)

                    # Check if the response is empty:
                    if not response:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # if the model refused to answer, we try again!
                    if ("sorry" in response_lc and (
                            "i cannot" in response_lc or "i can't" in response_lc or 'i am unable' in response_lc)) \
                            or "i cannot assist" in response_lc or "i can't assist" in response_lc or 'i am unable to assist' in response_lc or "I'm sorry" in response_lc:
                        aprint(
                            f"Model {model_name} refuses to assist (response: {response}). Trying again...")
                        continue
                    else:
                        return response

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback
                traceback.print_exc()
                return f"Error: '{e}'"

    def describe_audio(self,
                       audio_uri: str,
                       system: str = 'You are a helpful AI assistant that can describe/analyse audio.',
                       query: str = 'Here is an audio file, please carefully describe it in detail. If it is speach, please transcribe accurately.',
                       model_name: Optional[str] = None,
                       temperature: float = 0,
                       max_output_tokens: Optional[int] = None,
                       number_of_tries: int = 4,
                       ) -> str:

        """
        Describe an audio file using the model.

        Parameters
        ----------
        audio_uri: str
            Can be: a path to an audio file, a URL to an audio file, or a base64 encoded audio.
        system:
            System message to use
        query  : str
            Query to use with audio
        model_name   : str
            Model to use
        temperature: float
            Temperature to use
        max_output_tokens  : int
            Maximum number of tokens to use
        number_of_tries : int
            Number of times to try to send the request to the model

        Returns
        -------
        str
            Description of the audio

        """

        with (asection(
                f"Asking model {model_name} to describe a given audio: '{audio_uri}':")):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If no model is passed get a default model with image support:
            if model_name is None:
                model_name = self.get_best_model(require_audio=True)

            # If the model does not support audio, return an error:
            if not self.has_model_support_for(model_name=model_name,
                                              features=ModelFeatures.Audio):
                return f"Model '{model_name}' does not support audio."

            # if no max_output_tokens is passed, get the default value:
            if max_output_tokens is None:
                max_output_tokens = self.max_num_output_tokens(model_name)
            else:
                # Limit the number of tokens to the maximum allowed by the model:
                max_output_tokens = min(max_output_tokens,
                                        self.max_num_output_tokens(model_name))

            try:

                audio_model_name = self.get_best_model(ModelFeatures.Audio)

                # Retry in case of model refusing to answer:
                for tries in range(number_of_tries):

                    messages = []

                    # System message:
                    system_message = Message(role='system')
                    system_message.append_text(system)
                    messages.append(system_message)

                    # User message:
                    user_message = Message(role='user')
                    user_message.append_text(query)
                    user_message.append_audio_uri(audio_uri)

                    messages.append(user_message)

                    # Run agent:
                    response = self.generate_text_completion(messages=messages,
                                                             model_name=audio_model_name,
                                                             temperature=temperature,
                                                             max_output_tokens=max_output_tokens)

                    # Normalise response:
                    response = str(response)

                    # Check if the response is empty:
                    if not response:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    return response

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback
                traceback.print_exc()
                return f"Error: '{e}'"

    def describe_video(self,
                       video_uri: str,
                       system: str = 'You are a helpful AI assistant that can describe/analyse videos.',
                       query: str = 'Here is a video file, please carefully describe it in detail.',
                       model_name: Optional[str] = None,
                       temperature: float = 0,
                       max_output_tokens: Optional[int] = None,
                       number_of_tries: int = 4,
                       ) -> str:
        """
        Describe a video file using the model.

        Parameters
        ----------
        video_uri: str
            Can be: a path to a video file, a URL to a video file, or a base64 encoded video.
        system:
            System message to use
        query  : str
            Query to use with video
        model_name   : str
            Model to use
        temperature: float
            Temperature to use
        max_output_tokens  : int
            Maximum number of tokens to use
        number_of_tries : int
            Number of times to try to send the request to the model

        Returns
        -------
        str
            Description of the video

        """

        with (asection(
                f"Asking model {model_name} to describe a given video: '{video_uri}':")):
            aprint(f"Query: '{query}'")
            aprint(f"Model: '{model_name}'")
            aprint(f"Max tokens: '{max_output_tokens}'")

            # If no model is passed get a default model with video support:
            if model_name is None:
                model_name = self.get_best_model(require_video=True)

            # If the model does not support video, return an error:
            if not self.has_model_support_for(model_name=model_name,
                                              features=ModelFeatures.Video):
                return f"Model '{model_name}' does not support video."

            # if no max_output_tokens is passed, get the default value:
            if max_output_tokens is None:
                max_output_tokens = self.max_num_output_tokens(model_name)
            else:
                # Limit the number of tokens to the maximum allowed by the model:
                max_output_tokens = min(max_output_tokens,
                                        self.max_num_output_tokens(model_name))

            try:

                video_model_name = self.get_best_model(ModelFeatures.Video)

                # Retry in case of model refusing to answer:
                for tries in range(number_of_tries):

                    messages = []

                    # System message:
                    system_message = Message(role='system')
                    system_message.append_text(system)
                    messages.append(system_message)

                    # User message:
                    user_message = Message(role='user')
                    user_message.append_text(query)
                    user_message.append_video_uri(video_uri)

                    messages.append(user_message)

                    # Run agent:
                    response = self.generate_text_completion(messages=messages,
                                                             model_name=video_model_name,
                                                             temperature=temperature,
                                                             max_output_tokens=max_output_tokens)

                    # Normalize response:
                    response = str(response)

                    # Check if the response is empty:
                    if not response:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    # response in lower case and trimmed of white spaces
                    response_lc = response.lower().strip()

                    # Check if response is too short:
                    if len(response_lc) < 3:
                        aprint(f"Response is empty. Trying again...")
                        continue

                    return response

            except Exception as e:
                # Log the error:
                aprint(f"Error: '{e}'")
                # print stack trace:
                import traceback
                traceback.print_exc()
                return f"Error: '{e}'"
