"""Callback manager that dispatches API events to registered listeners.

This module provides ``ApiCallbackManager``, a composite implementation of
``BaseApiCallbacks`` that maintains a list of callback listeners and fans
out every API event (text generation, embedding, media description, etc.)
to all registered instances. It is used by ``BaseApi`` and its subclasses
to support pluggable monitoring and logging of API operations.
"""

from typing import Any, List, Sequence, Union

from litemind.agent.messages.message import Message
from litemind.apis.callbacks.base_api_callbacks import BaseApiCallbacks


class ApiCallbackManager(BaseApiCallbacks):
    """Manager that dispatches API callback events to registered listeners.

    Acts as a composite pattern: implements the same callback interface as
    ``BaseApiCallbacks`` and fans out each event to all registered callback
    instances.
    """

    def __init__(self):
        """Initialize with an empty list of callback listeners."""
        self.callbacks: List[BaseApiCallbacks] = []

    def add_callback(self, callback: BaseApiCallbacks) -> None:
        """Register a single callback listener.

        Parameters
        ----------
        callback : BaseApiCallbacks
            The callback instance to add. Duplicates are ignored.
        """
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def add_callbacks(
        self, callbacks_: Union[Sequence[BaseApiCallbacks], "ApiCallbackManager"]
    ) -> None:
        """Register multiple callback listeners at once.

        Parameters
        ----------
        callbacks_ : Union[Sequence[BaseApiCallbacks], ApiCallbackManager]
            A sequence of callback instances or another ``ApiCallbackManager``
            whose listeners will be merged into this manager.
        """
        if isinstance(callbacks_, ApiCallbackManager):
            self.callbacks.extend(callbacks_.callbacks)
        elif isinstance(callbacks_, Sequence):
            for callback in callbacks_:
                if isinstance(callback, BaseApiCallbacks):
                    self.callbacks.append(callback)

    def remove_callback(self, callback: BaseApiCallbacks) -> None:
        """Remove a previously registered callback listener.

        Parameters
        ----------
        callback : BaseApiCallbacks
            The callback instance to remove. Does nothing if not registered.
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def __contains__(self, callback: BaseApiCallbacks) -> bool:
        """Check whether a callback is registered in this manager."""
        return callback in self.callbacks

    def __len__(self) -> int:
        """Return the number of registered callback listeners."""
        return len(self.callbacks)

    def __getitem__(self, index: int) -> BaseApiCallbacks:
        """Return the callback at the given index."""
        return self.callbacks[index]

    def __setitem__(self, index: int, callback: BaseApiCallbacks) -> None:
        """Replace the callback at the given index."""
        self.callbacks[index] = callback

    def __delitem__(self, index: int) -> None:
        """Remove the callback at the given index."""
        del self.callbacks[index]

    def __iter__(self):
        """Iterate over all registered callback listeners."""
        return iter(self.callbacks)

    def on_availability_check(self, available: bool) -> None:
        """Dispatch availability check event to all listeners.

        Parameters
        ----------
        available : bool
            Whether the API is available.
        """
        for callback in self.callbacks:
            callback.on_availability_check(available)

    def on_model_list(self, models: List[str], **kwargs) -> None:
        """Dispatch model list retrieval event to all listeners.

        Parameters
        ----------
        models : List[str]
            The list of available model names.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_model_list(models, **kwargs)

    def on_best_model_selected(self, model_name: str, **kwargs) -> None:
        """Dispatch best model selection event to all listeners.

        Parameters
        ----------
        model_name : str
            The selected model name.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_best_model_selected(model_name, **kwargs)

    def on_text_generation(
        self, messages: List[Message], response: Message, **kwargs
    ) -> None:
        """Dispatch text generation event to all listeners.

        Parameters
        ----------
        messages : List[Message]
            The input messages sent to the model.
        response : Message
            The generated response message.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_text_generation(messages, response, **kwargs)

    def on_text_streaming(self, fragment: str, **kwargs) -> None:
        """Dispatch text streaming fragment event to all listeners.

        Parameters
        ----------
        fragment : str
            The streamed text fragment.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_text_streaming(fragment, **kwargs)

    def on_audio_transcription(
        self, transcription: str, audio_uri: str, **kwargs
    ) -> None:
        """Dispatch audio transcription event to all listeners.

        Parameters
        ----------
        transcription : str
            The transcribed text.
        audio_uri : str
            The URI of the transcribed audio.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_audio_transcription(transcription, audio_uri, **kwargs)

    def on_document_conversion(
        self, document_uri: str, markdown: str, **kwargs
    ) -> None:
        """Dispatch document conversion event to all listeners.

        Parameters
        ----------
        document_uri : str
            The URI of the converted document.
        markdown : str
            The resulting markdown content.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_document_conversion(document_uri, markdown, **kwargs)

    def on_video_conversion(
        self, video_uri: str, images: List[str], audio: str, **kwargs
    ) -> None:
        """Dispatch video conversion event to all listeners.

        Parameters
        ----------
        video_uri : str
            The URI of the converted video.
        images : List[str]
            The extracted image URIs.
        audio : str
            The extracted audio URI.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_video_conversion(video_uri, images, audio, **kwargs)

    def on_audio_generation(self, text: str, audio_uri: str, **kwargs) -> None:
        """Dispatch audio generation event to all listeners.

        Parameters
        ----------
        text : str
            The text that was converted to audio.
        audio_uri : str
            The URI of the generated audio.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_audio_generation(text, audio_uri, **kwargs)

    def on_image_generation(self, prompt: str, image: Any, **kwargs) -> None:
        """Dispatch image generation event to all listeners.

        Parameters
        ----------
        prompt : str
            The prompt used to generate the image.
        image : Any
            The generated image object.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_image_generation(prompt, image, **kwargs)

    def on_text_embedding(
        self, texts: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """Dispatch text embedding event to all listeners.

        Parameters
        ----------
        texts : List[str]
            The input texts that were embedded.
        embeddings : Sequence[Sequence[float]]
            The resulting embedding vectors.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_text_embedding(texts, embeddings, **kwargs)

    def on_image_embedding(
        self, image_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """Dispatch image embedding event to all listeners.

        Parameters
        ----------
        image_uris : List[str]
            The input image URIs that were embedded.
        embeddings : Sequence[Sequence[float]]
            The resulting embedding vectors.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_image_embedding(image_uris, embeddings, **kwargs)

    def on_audio_embedding(
        self, audio_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """Dispatch audio embedding event to all listeners.

        Parameters
        ----------
        audio_uris : List[str]
            The input audio URIs that were embedded.
        embeddings : Sequence[Sequence[float]]
            The resulting embedding vectors.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_audio_embedding(audio_uris, embeddings, **kwargs)

    def on_video_embedding(
        self, video_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """Dispatch video embedding event to all listeners.

        Parameters
        ----------
        video_uris : List[str]
            The input video URIs that were embedded.
        embeddings : Sequence[Sequence[float]]
            The resulting embedding vectors.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_video_embedding(video_uris, embeddings, **kwargs)

    def on_document_embedding(
        self, document_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        """Dispatch document embedding event to all listeners.

        Parameters
        ----------
        document_uris : List[str]
            The input document URIs that were embedded.
        embeddings : Sequence[Sequence[float]]
            The resulting embedding vectors.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_document_embedding(document_uris, embeddings, **kwargs)

    def on_image_description(self, image_uri: str, description: str, **kwargs) -> None:
        """Dispatch image description event to all listeners.

        Parameters
        ----------
        image_uri : str
            The URI of the described image.
        description : str
            The generated description.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_image_description(image_uri, description, **kwargs)

    def on_audio_description(self, audio_uri: str, description: str, **kwargs) -> None:
        """Dispatch audio description event to all listeners.

        Parameters
        ----------
        audio_uri : str
            The URI of the described audio.
        description : str
            The generated description.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_audio_description(audio_uri, description, **kwargs)

    def on_video_description(self, video_uri: str, description: str, **kwargs) -> None:
        """Dispatch video description event to all listeners.

        Parameters
        ----------
        video_uri : str
            The URI of the described video.
        description : str
            The generated description.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_video_description(video_uri, description, **kwargs)

    def on_document_description(
        self, document_uri: str, description: str, **kwargs
    ) -> None:
        """Dispatch document description event to all listeners.

        Parameters
        ----------
        document_uri : str
            The URI of the described document.
        description : str
            The generated description.
        **kwargs
            Additional keyword arguments forwarded to listeners.
        """
        for callback in self.callbacks:
            callback.on_document_description(document_uri, description, **kwargs)
