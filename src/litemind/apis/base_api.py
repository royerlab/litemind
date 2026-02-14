"""Abstract base class defining the interface for all LLM provider APIs.

This module declares ``BaseApi``, the top-level abstract class that every
provider implementation (OpenAI, Anthropic, Gemini, Ollama) must inherit.
It specifies the full contract for text generation, media embedding, audio
transcription, image/audio/video/document description, and credential
verification.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Type, Union

from PIL.Image import Image
from pydantic import BaseModel

from litemind.agent.messages.message import Message
from litemind.agent.tools.toolset import ToolSet
from litemind.apis.callbacks.api_callback_manager import ApiCallbackManager
from litemind.apis.model_features import ModelFeatures
from litemind.media.media_base import MediaBase
from litemind.utils.random_projector import DeterministicRandomProjector


class BaseApi(ABC):
    """Abstract base class defining the interface for all LLM provider APIs.

    All provider implementations (OpenAI, Anthropic, Gemini, Ollama) must
    inherit from this class and implement its abstract methods for text
    generation, media embedding, audio transcription, and media description.
    """

    def __init__(self, callback_manager: Optional[ApiCallbackManager] = None, **kwargs):
        """Initialize the base API with an optional callback manager.

        Parameters
        ----------
        callback_manager : Optional[ApiCallbackManager]
            Callback manager for monitoring API operations. If None, a default
            (no-op) manager is created.
        """
        if callback_manager is not None:
            self.callback_manager = callback_manager
        else:
            self.callback_manager = ApiCallbackManager()

    @abstractmethod
    def check_availability_and_credentials(
        self, api_key: Optional[str] = None
    ) -> Optional[bool]:
        """Check if this API is available and whether credentials are valid.

        If no API key is provided, checks the key available in the
        environment. If no API key is required, returns None.

        Parameters
        ----------
        api_key : Optional[str]
            The API key to check. If None, checks environment variables.

        Returns
        -------
        Optional[bool]
            True if available and valid, False if unavailable, None if no
            credentials are required.

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
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
    ) -> List[str]:
        """List models that satisfy the given capability requirements.

        Filters models so that each returned model supports all requested
        *features*, none of the *non_features*, and all *media_types*.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required feature(s) to filter on.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Feature(s) to exclude from the list.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must support (e.g., Text, Image, Audio).

        Returns
        -------
        List[str]
            List of matching model names.

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
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        exclusion_filters: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Get the best model that satisfies the given capability requirements.

        Returns the highest-ranked model that supports all requested
        *features*, none of the *non_features*, and all *media_types*.

        Parameters
        ----------
        features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Required features the model must support.
        non_features : Optional[Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]]
            Features the model must not have.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must support.
        exclusion_filters : Optional[List[str]]
            Substrings that, if found in a model name, exclude it.

        Returns
        -------
        Optional[str]
            The best matching model name, or None if no model matches.

        """
        pass

    @abstractmethod
    def has_model_support_for(
        self,
        features: Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]],
        media_types: Optional[Sequence[Type[MediaBase]]] = None,
        model_name: Optional[str] = None,
    ) -> bool:
        """Check if the given model supports the given features and media types.

        If no model is provided, the default (and ideally best) model is used.
        In that case this method returns True if any model supports the
        requested features and media types.

        Notes
        -----
        The difference between requesting a model that supports the
        ``Audio`` feature versus requesting a model that can receive an
        ``Audio`` media is subtle but important: the ``Audio`` feature
        means that the model supports audio *natively*.  However,
        litemind's automatic media conversion can handle audio files
        that carry voice — for example, via Whisper transcription.  In
        that case, the model can carry ``Audio`` media in messages but
        does not support the ``Audio`` feature.  You can check for
        audio *conversion* support with
        ``ModelFeatures.AudioConversion``.

        Parameters
        ----------
        features : Union[str, List[str], ModelFeatures, Sequence[ModelFeatures]]
            The feature(s) to check for.
        media_types : Optional[Sequence[Type[MediaBase]]]
            Media types the model must support.
        model_name : Optional[str]
            The model to check. If None, checks any available model.

        Returns
        -------
        bool
            True if the model supports the requested features and media types.

        """
        pass

    @abstractmethod
    def get_model_features(self, model_name: str) -> List[ModelFeatures]:
        """Get the features of the given model.

        Parameters
        ----------
        model_name : str
            The model name to query.

        Returns
        -------
        List[ModelFeatures]
            All features the model supports.

        """
        pass

    @abstractmethod
    def max_num_input_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of input tokens for the given model.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query. If None, uses the best available model.

        Returns
        -------
        int
            The maximum number of input tokens for the model.

        """
        pass

    @abstractmethod
    def max_num_output_tokens(self, model_name: Optional[str] = None) -> int:
        """Get the maximum number of output tokens for the given model.

        Parameters
        ----------
        model_name : Optional[str]
            The model to query. If None, uses the best available model.

        Returns
        -------
        int
            The maximum number of output tokens for the model.

        """
        pass

    @abstractmethod
    def count_tokens(self, text: str, model_name: Optional[str] = None) -> int:
        """Count the number of tokens in the given text.

        Parameters
        ----------
        text : str
            The text to count the tokens in.
        model_name : Optional[str]
            The model whose tokenizer to use.

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
        """Generate text from the given messages using the specified model.

        Parameters
        ----------
        messages : List[Message]
            The list of messages to send to the model.
        model_name : Optional[str]
            The model to use. If None, uses the best available model.
        temperature : float
            Sampling temperature (0.0 = deterministic).
        max_num_output_tokens : Optional[int]
            Maximum tokens in the response.
        toolset : Optional[ToolSet]
            The toolset available for function calling.
        use_tools : bool
            If True, executes tool calls automatically until a final
            answer is reached. If False, returns raw ToolCall blocks.
        response_format : Optional[BaseModel]
            Pydantic model for structured output parsing.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        List[Message]
            Response messages. For simple requests this is a single Message;
            when tools are used, includes intermediate tool call/result messages.

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
        """Generate an audio file from text using the model.

        Parameters
        ----------
        text : str
            The text to convert to audio.
        voice : Optional[str]
            The voice to use for synthesis.
        audio_format : Optional[str]
            Audio file format: "mp3", "flac", "wav", or "pcm". Defaults to "mp3".
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        **kwargs
            Additional provider-specific arguments.

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
        """Generate an image using the default or given model.

        Parameters
        ----------
        positive_prompt : str
            The prompt describing what to generate.
        negative_prompt : Optional[str]
            Prompt specifying what not to generate.
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        image_width : int
            Desired width of the image in pixels.
        image_height : int
            Desired height of the image in pixels.
        preserve_aspect_ratio : bool
            Whether to preserve the aspect ratio of the image.
        allow_resizing : bool
            Whether to allow resizing to match model constraints.
        **kwargs
            Additional provider-specific arguments.

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
        """Generate a video using the model.

        Parameters
        ----------
        description : str
            Text description of the video to generate.
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        **kwargs
            Additional provider-specific arguments.

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
        """Embed text using the model.

        Parameters
        ----------
        texts : Sequence[str]
            The text snippets to embed.
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input text.

        """

        pass

    def _reduce_embeddings_dimension(
        self, embeddings: Sequence[Sequence[float]], reduced_dim: int
    ) -> Sequence[Sequence[float]]:
        """Reduce embedding dimensionality using a deterministic random projection.

        Parameters
        ----------
        embeddings : Sequence[Sequence[float]]
            The embeddings to reduce.
        reduced_dim : int
            The target dimensionality.

        Returns
        -------
        Sequence[Sequence[float]]
            The reduced-dimensionality embeddings.
        """
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
        """Embed images using the given model.

        Parameters
        ----------
        image_uris : List[str]
            List of image URIs or file paths.
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input image.

        """

    @abstractmethod
    def embed_audios(
        self,
        audio_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Embed audio files using the given model.

        Parameters
        ----------
        audio_uris : List[str]
            List of audio URIs or file paths.
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input audio.

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
        """Embed videos using the given model.

        Parameters
        ----------
        video_uris : List[str]
            List of video URIs or file paths.
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input video.

        """

        pass

    @abstractmethod
    def embed_documents(
        self,
        document_uris: List[str],
        model_name: Optional[str] = None,
        dimensions: int = 512,
        **kwargs,
    ) -> Sequence[Sequence[float]]:
        """Embed documents using the given model.

        Parameters
        ----------
        document_uris : List[str]
            List of document URIs or file paths.
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        dimensions : int
            Target dimensionality for the embedding vectors.
        **kwargs
            Additional provider-specific arguments.

        Returns
        -------
        Sequence[Sequence[float]]
            Embedding vectors, one per input document.

        """

        pass

    @abstractmethod
    def transcribe_audio(
        self, audio_uri: str, model_name: Optional[str] = None, **model_kwargs
    ) -> str:
        """Transcribe an audio file using the model.

        Parameters
        ----------
        audio_uri : str
            Path, URL, or base64-encoded audio to transcribe.
        model_name : Optional[str]
            The model to use. If None, selects the best available.
        **model_kwargs
            Additional provider-specific arguments.

        Returns
        -------
        str
            The transcription of the audio.

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
    ) -> Optional[str]:
        """Describe an image using the model.

        Parameters
        ----------
        image_uri : str
            Path, URL, or base64-encoded image to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the image.
        model_name : Optional[str]
            Model to use. If None, selects the best available.
        temperature : float
            Sampling temperature for the response.
        max_output_tokens : Optional[int]
            Maximum tokens in the response.
        number_of_tries : int
            Number of retry attempts on failure.

        Returns
        -------
        Optional[str]
            Text description of the image, or None if all attempts fail.

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
        """Describe an audio file using the model.

        Parameters
        ----------
        audio_uri : str
            Path, URL, or base64-encoded audio to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the audio.
        model_name : Optional[str]
            Model to use.
        temperature : float
            Temperature to use.
        max_output_tokens : Optional[int]
            Maximum number of tokens to use.
        number_of_tries : int
            Number of times to try to send the request to the model.

        Returns
        -------
        Optional[str]
            Description of the audio, or None if all attempts fail.

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
        """Describe a video file using the model.

        Parameters
        ----------
        video_uri : str
            Path, URL, or base64-encoded video to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the video.
        model_name : Optional[str]
            Model to use.
        temperature : float
            Temperature to use.
        max_output_tokens : Optional[int]
            Maximum number of tokens to use.
        number_of_tries : int
            Number of times to try to send the request to the model.

        Returns
        -------
        Optional[str]
            Description of the video, or None if all attempts fail.

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
        """Describe a document using the model.

        Parameters
        ----------
        document_uri : str
            Path, URL, or base64-encoded document to describe.
        system : str
            System message to guide the description.
        query : str
            Query prompt to use with the document.
        model_name : Optional[str]
            Model to use.
        temperature : float
            Temperature to use.
        max_output_tokens : Optional[int]
            Maximum number of tokens to use.
        number_of_tries : int
            Number of times to try to send the request to the model.

        Returns
        -------
        Optional[str]
            Description of the document, or None if all attempts fail.

        """

        pass
