from abc import ABC
from typing import Any, List, Sequence

from litemind.agent.messages.message import Message


class BaseApiCallbacks(ABC):
    def on_availability_check(self, available: bool) -> None:
        """
        Called when the availability and credentials check is performed.

        Parameters
        ----------
        available: bool
            Whether the API is available.

        """
        pass

    def on_model_list(self, models: List[str], **kwargs) -> None:
        """
        Called when the list of models is retrieved.

        Parameters
        ----------

        models: List[str]
            The list of models.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_best_model_selected(self, model_name: str, **kwargs) -> None:
        """
        Called when the best model is selected.

        Parameters
        ----------

        model_name: str
            The model name.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_text_generation(
        self, messages: List[Message], response: Message, **kwargs
    ) -> None:
        """
        Called when a text completion is generated.

        Parameters
        ----------
        response: Message
            The response message.
        messages: List[Message]
            The list of messages.
        kwargs
            Other and additional keyword arguments.
        """
        pass

    def on_text_streaming(self, fragment: str, **kwargs) -> None:
        """
        Called when a text completion is generated.

        Parameters
        ----------
        fragment: str
            Streaming fragment during generation.
        kwargs
            Other and additional keyword arguments.
        """
        pass

    def on_audio_transcription(
        self, transcription: str, audio_uri: str, **kwargs
    ) -> None:
        """
        Called when an audio transcription is generated.

        Parameters
        ----------

        transcription: str
            The transcription.
        audio_uri: str
            The audio URI.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_document_conversion(
        self, document_uri: str, markdown: str, **kwargs
    ) -> None:
        """
        Called when a document is converted to markdown.

        Parameters
        ----------

        document_uri: str
            The document URI.
        markdown: str
            The markdown content.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_video_conversion(
        self, video_uri: str, images: List[str], audio: str, **kwargs
    ) -> None:
        """
        Called when a video is converted to images and audio.

        Parameters
        ----------

        video_uri: str
            The video URI.
        images: List[str]
            The list of image URIs.
        audio: str
            The audio URI.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_audio_generation(self, text: str, audio_uri: str, **kwargs) -> None:
        """
        Called when an audio file is generated.

        Parameters
        ----------

        text: str
            The text prompt.
        audio_uri: str
            The audio URI.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_image_generation(self, prompt: str, image: Any, **kwargs) -> None:
        """
        Called when an image is generated.

        Parameters
        ----------
        prompt: str
            The image generation prompt.
        image: Any
            The generated image.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_text_embedding(
        self, texts: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """
        Called when text embeddings are generated.

        Parameters
        ----------
        texts: List[str]
            The input texts.
        embeddings: Sequence[Sequence[float]]
            The generated embeddings.
        kwargs: dict
            Other and additional keyword arguments.


        """
        pass

    def on_image_embedding(
        self, image_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """
        Called when image embeddings are generated.

        Parameters
        ----------
        image_uris: List[str]
            The input image URIs.
        embeddings: Sequence[Sequence[float]]
            The generated embeddings.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_audio_embedding(
        self, audio_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """
        Called when audio embeddings are generated.

        Parameters
        ----------
        audio_uris: List[str]
            The input audio URIs.
        embeddings: Sequence[Sequence[float]]
            The generated embeddings.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_video_embedding(
        self, video_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """
        Called when video embeddings are generated.

        Parameters
        ----------
        video_uris: List[str]
            The input video URIs.
        embeddings: Sequence[Sequence[float]]
            The generated embeddings.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_document_embedding(
        self, document_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """
        Called when document embeddings are generated.

        Parameters
        ----------
        document_uris: List[str]
            The input video URIs.
        embeddings: Sequence[Sequence[float]]
            The generated embeddings.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_image_description(self, image_uri: str, description: str, **kwargs) -> None:
        """
        Called when an image is described.

        Parameters
        ----------

        image_uri: str
            The image URI.
        description: str
            The description.
        kwargs: dict
            Other and additional keyword arguments.

        """
        pass

    def on_audio_description(self, audio_uri: str, description: str, **kwargs) -> None:
        """
        Called when an audio is described.

        Parameters
        ----------

        audio_uri: str
            The audio URI.
        description: str
            The description.
        kwargs: dict
            Other and additional keyword arguments.
        """
        pass

    def on_video_description(self, video_uri: str, description: str, **kwargs) -> None:
        """
        Called when a video is described.

        Parameters
        ----------
        video_uri: str
            The video URI.
        description: str
            The description.
        kwargs: dict
            Other and additional keyword arguments.
        """
        pass

    def on_document_description(
        self, document_uri: str, description: str, **kwargs
    ) -> None:
        """
        Called when a document is described.

        Parameters
        ----------
        document_uri: str
            The video URI.
        description: str
            The description.
        kwargs: dict
            Other and additional keyword arguments.
        """
        pass
