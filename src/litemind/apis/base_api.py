from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Union

from PIL.Image import Image
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.callbacks.callback_manager import CallbackManager
from litemind.apis.model_features import ModelFeatures
from litemind.utils.random_projector import DeterministicRandomProjector


class BaseApi(ABC):
    """
    The base class for all APIs. This class provides the interface for all provider APIs to implement.
    """

    # constructor:
    def __init__(self, callback_manager: Optional[CallbackManager] = None):

        if callback_manager is not None:
            self.callback_manager = callback_manager
        else:
            self.callback_manager = CallbackManager()

    @abstractmethod
    def check_availability_and_credentials(
        self, api_key: Optional[str] = None
    ) -> Optional[bool]:
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
    def list_models(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
    ) -> List[str]:
        """
        Get the list of models available that satisfy a given set of included and excluded features.

        Parameters
        ----------
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            The feature(s) to filter on.
        non_features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            The feature(s) to exclude from the list.

        Returns
        -------
        List[str]
            The list of models available

        """
        pass

    @abstractmethod
    def get_best_model(
        self,
        features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        non_features: Optional[
            Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
        ] = None,
        exclusion_filters: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Get the name of the best possible model that satisfies a set of capability requirements.

        Parameters
        ----------
        features: Sequence[ModelFeatures], or ModelFeatures, or str, or List[str]
            List of features to filter on.
        non_features: Sequence[ModelFeatures], or ModelFeatures, or str, or List[str]
            List of features to exclude from the list.
        exclusion_filters: Optional[List[str]]
            List of strings that if found in the model name exclude it.


        Returns
        -------
        str
            The default model to use.
            None if no models satisfy the requirements

        """
        pass

    @abstractmethod
    def has_model_support_for(
        self,
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]],
        model_name: Optional[str] = None,
    ) -> bool:
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
        pass

    @abstractmethod
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
        pass

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
    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """
        Count the number of tokens in the given text.

        Parameters
        ----------
        text: str
            The text to count the tokens in.
        model_name: Optional[str]
            The name of the model to use.

        Returns
        -------
        int
            The number of tokens in the text.

        """
        pass

    @abstractmethod
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
        max_num_output_tokens: Optional[int]
            The maximum number of tokens to use.
        toolset: Optional[ToolSet]
            The toolset to use.
        use_tools: bool
            Whether to use tools. If True, the toolset must be provided.
            In that case this method will run use the tools that the model deems necessary until reaching a final answer.
            If False, when the text generation model requests tool use, the tools are not used and the response contains a message with a ToolCall block.
        response_format: Optional[BaseModel | str]
            The response format to use. Provide a pydantic object to use as the output format or json schema.
        kwargs: dict
            Additional arguments to pass to the completion function, this is API specific!


        Returns
        -------
        List[Message]
            The response from the model. A response is a list of messages.
            For simple requests this is usually a single Message,
            but when using tools there is typically multiple messages,
            including user messages for the tool responses.

        """
        pass

    @abstractmethod
    def transcribe_audio(
        self, audio_uri: str, model_name: Optional[str] = None, **kwargs
    ) -> str:
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
        pass

    @abstractmethod
    def convert_audio_in_messages(self, messages: List[Message]) -> List[Message]:
        """
        Convert audio in messages into text.

        Parameters
        ----------
        messages: List[Message]
            The list of Message objects to process.

        Returns
        -------
        List[Message]
            The list of Message objects with audio converted to text.

        """

    @abstractmethod
    def convert_documents_to_markdown_in_messages(
        self,
        messages: List[Message],
        exclude_extensions: Optional[List[str]] = None,
        model_name: Optional[str] = None,
    ) -> List[Message]:
        """
        Convert documents in messages into text in Markdown format.

        Parameters
        ----------
        messages : List[Message]
            The list of Message objects to process.
        exclude_extensions : Optional[List[str]]
            The list of file extensions to exclude from processing.
        model_name : Optional[str]
            The name of the model to use.


        Returns
        -------
        List[Message]
            The list of Message objects with documents converted to markdown.
        """

    pass

    @abstractmethod
    def convert_videos_to_images_and_audio_in_messages(
        self, message: Sequence[Message]
    ) -> List[Message]:
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
        pass

    @abstractmethod
    def generate_audio(
        self,
        text: str,
        voice: Optional[str] = None,
        audio_format: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Generate an audio file using the model.

        Parameters
        ----------
        text: str
            The text to convert to audio.
        voice: Optional[str]
            The voice to use.
        audio_format: Optional[str]
            The format of the audio file. Can be: "mp3", "flac", "wav", or "pcm". By default, "mp3" is used.
        model_name: Optional[str]
            The name of the model to use.
        kwargs: dict
            Additional arguments to pass to the audio generation function.

        Returns
        -------
        str
            URI of the audio file.

        """
        pass

    @abstractmethod
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
        """
        Generate an image using the default or given model.

        Parameters
        ----------
        positive_prompt: str
            The prompt to use for image generation.
        negative_prompt: str
            Prompt to specifying what not to generate
        model_name: str
            The name of the model to use.
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
        pass

    @abstractmethod
    def generate_video(
        self, description: str, model_name: Optional[str] = None, **kwargs
    ) -> str:
        """
        Generate a video using the model.

        Parameters
        ----------
        description: str
            The description of the video.
        model_name: Optional[str]
            The name of the model to use.
        kwargs: dict
            Additional arguments to pass to the video generation function.

        Returns
        -------
        str
            URI of the generated video.

        """

        pass

    def embed_texts(
        self,
        texts: Sequence[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
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

        pass

    def _reduce_embeddings_dimension(
        self, embeddings: Sequence[Sequence[float]], reduced_dim: int
    ) -> Sequence[Sequence[float]]:

        # Create a DeterministicRandomProjector object:
        drp = DeterministicRandomProjector(
            original_dim=len(embeddings[0]), reduced_dim=reduced_dim
        )

        # Project the embeddings:
        return drp.transform(embeddings)

    @abstractmethod
    def embed_images(
        self,
        image_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """
        Embed images using the given model.

        Parameters
        ----------
        image_uris: List[str]
            List of image URIs.
        model_name: Optional[str]
            The name of the model to use.
        dimensions: int
            The number of dimensions for the embedding.
        kwargs: dict
            Additional arguments to pass to the embedding function.

        Returns
        -------
        Sequence[Sequence[float]]
            The embeddings of the images.

        """

    @abstractmethod
    def embed_audios(
        self,
        audio_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """
        Embed audios using the given model.

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

        pass

    @abstractmethod
    def embed_videos(
        self,
        video_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """
        Embed videos using the given model.

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

        pass

    @abstractmethod
    def embed_documents(
        self,
        documents_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """
        Embed documents using the given model.

        Parameters
        ----------
        documents_uris: List[str]
            List of document URIs.
        model_name: Optional[str]
            The name of the model to use.
        dimensions: int
            The number of dimensions for the embedding.
        kwargs: dict
            Additional arguments to pass to the embedding function.

        Returns
        -------
        Sequence[Sequence[float]]
            The embeddings of the documents.

        """

        pass

    @abstractmethod
    def describe_image(
        self,
        image_uri: str,
        system: str = "You are a helpful AI assistant that can describe and analyse images.",
        query: str = "Here is an image, please carefully describe it completely and in detail.",
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

        pass

    @abstractmethod
    def describe_audio(
        self,
        audio_uri: str,
        system: str = "You are a helpful AI assistant that can describe and analyse audio.",
        query: str = "Here is an audio file, please carefully describe its contents in detail. If it is speech, please transcribe completely and accurately.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> Optional[str]:
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

    @abstractmethod
    def describe_video(
        self,
        video_uri: str,
        system: str = "You are a helpful AI assistant that can describe/analyse videos.",
        query: str = "Here is a video file, please carefully and completely describe it in detail.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> Optional[str]:
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
        str | None
            Description of the video

        """

        pass

    @abstractmethod
    def describe_document(
        self,
        document_uri: str,
        system: str = "You are a helpful AI assistant that can describe/analyse documents.",
        query: str = "Here is a document, please carefully and completely describe it in detail.",
        model_name: Optional[str] = None,
        temperature: float = 0,
        max_output_tokens: Optional[int] = None,
        number_of_tries: int = 4,
    ) -> Optional[str]:
        """
        Describe a document using the model.

        Parameters
        ----------
        document_uri: str
            Can be: a path to a document file, a URL to a document file, or a base64 encoded document.
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
        str | None
            Description of the document

        """

        pass
