from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from PIL.Image import Image
from arbol import asection, aprint
from pandas import DataFrame
from pydantic import BaseModel

from litemind.agent.message import Message
from litemind.agent.message_block_type import BlockType
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.model_features import ModelFeatures
from litemind.apis.utils.document_processing import is_pymupdf_available, \
    convert_document_to_markdown, extract_images_from_document
from litemind.apis.utils.fastembed_embeddings import is_fastembed_available, \
    fastembed_text
from litemind.apis.utils.random_projector import DeterministicRandomProjector
from litemind.apis.utils.transform_video_uris_to_images_and_audio import \
    transform_video_uris_to_images_and_audio
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
                       exclusion_filters: Optional[List[str]] = None
                       ) -> Optional[str]:
        """
        Get the name of the best possible model that satisfies a set of capability requirements.

        Parameters
        ----------
        features: Sequence[ModelFeatures], or ModelFeatures, or str, or List[str]
            List of features to filter on.
        exclusion_filters: Optional[List[str]]
            List of strings that if found in the model name exclude it.


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
                           ModelFeatures]],
                       exclusion_filters: Optional[Union[str,List[str]]] = None) -> List[str]:
        """
        Filter the list of models based on the given features.
        Parameters
        ----------
        model_list: List[str]
            List of models.
        features: Sequence[ModelFeatures], or ModelFeatures, or str, or List[str]
            List of features to filter on.
        exclusion_filters: Optional[Union[str,List[str]]]
            List of strings that if found in the model name exclude it.

        Returns
        -------
        List[str]
            Filtered list of models.

        """
        filtered_model_list = []

        if not features:
            # Nothing to filter:
            return model_list

        # if exclusion_filters is a string then wrap in singleton list:
        if isinstance(exclusion_filters, str):
            exclusion_filters = [exclusion_filters]

        for model in model_list:

            # Check if the model should be excluded:
            if exclusion_filters and any(
                    [filter in model for filter in exclusion_filters]):
                continue

            # Append models that support the given features:
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

            # t=The following features use 'fallback' best-of-kind libraries:

            if feature == ModelFeatures.AudioTranscription:
                # We provide a default implementation for audio transcription using local whisper is available:
                if not is_local_whisper_available():
                    return False

            elif feature == ModelFeatures.Documents:
                # Checks if document processing library is installed:
                if not is_pymupdf_available():
                    return False

            elif feature == ModelFeatures.TextEmbeddings:
                # Check if the fastembed library is installed:
                if not is_fastembed_available():
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
                                 response_format: Optional[BaseModel] = None,
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
        response_format: Optional[BaseModel | str]
            The response format to use. Provide a pydantic object to use as the output format or json schema.
        kwargs: dict
            Additional arguments to pass to the completion function, this is API specific!


        Returns
        -------
        Message
            The response from the model.

        """
        pass

    def transcribe_audio(self,
                         audio_uri: str,
                         model_name: Optional[str] = None,
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

            # Iterate over each block in the message:
            for block in message.blocks:
                if block.block_type == BlockType.Audio and block.content:
                    try:
                        # We use a local instance of whisper instead:
                        transcription = self.transcribe_audio(block.content)

                        # Extract filename from URI:
                        original_filename = block.content.split("/")[-1]

                        # Add markdown quotes ''' around the transcribed text, and
                        # add prefix: "Transcription: " to the transcribed text:
                        transcription = f"\nTranscription of audio file '{original_filename}': \n'''\n{transcription}\n'''\n"

                        # Add the transcribed text to the message
                        message.append_text(transcription)

                        # Remove the audio block:
                        message.blocks.remove(block)

                    except Exception as e:
                        raise ValueError(
                            f"Could not transcribe audio from: '{block.content}', error: {e}")

        return messages

    def _convert_documents_to_markdown_in_messages(self,
                                                   messages: List[Message]) -> \
            List[Message]:
        """
        Convert documents in messages into text in markdown format.

        Parameters
        ----------
        messages : List[Message]
            The list of Message objects to process.

        Returns
        -------
        List[Message]
            The list of Message objects with documents converted to markdown.
        """

        converted_messages = []

        # Iterate over each message in the list:
        for message in messages:

            # Create a new message object:
            converted_message = Message(role=message.role)

            # Add the new message to the list:
            converted_messages.append(converted_message)

            # Iterate over each block in the message:
            for block in message.blocks:
                if block.block_type == BlockType.Document and block.content is not None:
                    try:
                        # Extract filename from URI:
                        original_filename = block.content.split("/")[-1]

                        # Extract the file extension from URI:
                        file_extension = block.content.split(".")[-1]

                        # Convert document to markdown:
                        markdown = convert_document_to_markdown(block.content)

                        # Add markdown quotes around the text, and add the extension, for example: ```pdf for extension 'pdf':
                        text = f"\nText extracted from document '{original_filename}': \n'''{file_extension}\n{markdown}\n'''\n"

                        # Add the extracted text to the message
                        converted_message.append_text(text)

                        # Extract images from document:
                        image_uris = extract_images_from_document(block.content)

                        # Add the extracted images to the message:
                        for image_uri in image_uris:
                            converted_message.append_image(image_uri)


                    except Exception as e:
                        raise ValueError(
                            f"Could not extract text from document: '{block.content}', error: {e}")
                elif block.block_type == BlockType.Json and block.content is not None:
                    try:
                        # Convert JSON to markdown:
                        markdown = f"```json\n{block.content}\n```"

                        # Add the markdown to the message:
                        converted_message.append_text(markdown)

                    except Exception as e:
                        raise ValueError(
                            f"Could not convert JSON to markdown: '{block.content}', error: {e}")
                elif block.block_type == BlockType.Object and block.content is not None:
                    try:
                        # Convert object to json since it is a pydantic object:
                        json_str = block.content.model_dump_json()
                        markdown = f"```json\n{json_str}\n```"

                        # Add the markdown to the message:
                        converted_message.append_text(markdown)
                    except Exception as e:
                        raise ValueError(
                            f"Could not convert object to markdown: '{block.content}', error: {e}")
                elif block.block_type == BlockType.Code and block.content is not None:
                    try:
                        # Get language from block attributes with default '':
                        language = block.attributes.get('language', '')

                        # Convert code to markdown:
                        markdown = f"```{language}\n{block.content}\n```"

                        # Add the markdown to the message:
                        converted_message.append_text(markdown)
                    except Exception as e:
                        raise ValueError(
                            f"Could not convert code to markdown: '{block.content}', error: {e}")
                elif block.block_type == BlockType.Table and block.content is not None:
                    try:
                        table: DataFrame = block.content

                        # pretty print the pandas dataframe into markdown-like / compatible string:
                        markdown_table = table.to_markdown()

                        # Convert table to markdown:
                        markdown = f"```dataframe\n{markdown_table}\n```"

                        # Add the markdown to the message:
                        converted_message.append_text(markdown)
                    except Exception as e:
                        raise ValueError(
                            f"Could not convert table to markdown: '{block.content}', error: {e}")
                elif block.block_type == BlockType.Text and block.content is not None:
                    try:
                        text: str = block.content

                        # Check if the text ends with at least one '\n' if not, then add it:
                        if not text.endswith('\n'):
                            text += '\n'

                        # Add the text to the message:
                        converted_message.append_text(text)

                    except Exception as e:
                        raise ValueError(
                            f"Could not convert table to markdown: '{block.content}', error: {e}")
                elif (block.block_type == BlockType.Image or
                      block.block_type == BlockType.Audio or
                      block.block_type == BlockType.Video):

                    # First we make a copy of the block:
                    block = block.copy()

                    # Add the block to the message:
                    converted_message.blocks.append(block)

                else:
                    raise ValueError(
                        f"Block type '{block.block_type}' not supported for conversion to markdown.")

        return converted_messages

    def _convert_videos_to_images_and_audio(self, message: Sequence[Message]) -> \
            List[Message]:

        """
        Convert videos in the message to images and audio blocks.

        Parameters
        ----------
        message : Sequence[Message]
            The message object containing video URIs.

        Returns
        -------
        List[Message]
            The message object with video URIs converted to images and audio blocks.
        """
        return [transform_video_uris_to_images_and_audio(msg) for msg in
                message]

    def generate_audio(self,
                       text: str,
                       voice: Optional[str] = None,
                       audio_format: Optional[str] = None,
                       model_name: Optional[str] = None,
                       **kwargs) -> str:
        """
        Generate an audio file using the model.

        Parameters
        ----------
        text: str
            The text to convert to audio.
        voice: Optional[str]
            The voice to use.
        audio_format: Optional[str]
            The format of the audio file. Can be: "mp3", "flac", "wav", or "pcm". By default "mp3" is used.
        model_name: Optional[str]
            The name of the model to use.
        kwargs: dict
            Additional arguments to pass to the audio generation function.

        Returns
        -------
        str
            URI of the audio file.

        """

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

        if is_fastembed_available():
            return fastembed_text(texts=texts,
                                  dimensions=dimensions,
                                  **kwargs)
        else:
            raise NotImplementedError(
                "Text embedding is not supported by this API.")

    def _reduce_embdeddings_dimension(self,
                                      embeddings: Sequence[Sequence[float]],
                                      reduced_dim: int) -> Sequence[
        Sequence[float]]:

        # Create a DeterministicRandomProjector object:
        drp = DeterministicRandomProjector(original_dim=len(embeddings[0]),
                                           reduced_dim=reduced_dim)

        # Project the embeddings:
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

        # If no model is passed get a default model with video embedding support:
        if model_name is None:
            model_name = self.get_best_model(features=[ModelFeatures.VideoEmbeddings])

        # Model must support videos:
        if model_name is None or not self.has_model_support_for(
                model_name=model_name, features=ModelFeatures.VideoEmbeddings):
            raise ValueError(f"Model '{model_name}' does not support video embedding.")

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
        video_embeddings = self.embed_texts(texts=video_descriptions,
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
                    user_message.append_image(image_uri)

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
                    user_message.append_audio(audio_uri)

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
                    user_message.append_video(video_uri)

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
