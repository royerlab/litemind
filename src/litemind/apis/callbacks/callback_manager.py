from typing import Any, List, Sequence, Union

from litemind.agent.messages.message import Message
from litemind.apis.callbacks.base_callbacks import BaseCallbacks


class CallbackManager(BaseCallbacks):
    def __init__(self):
        self.callbacks: List[BaseCallbacks] = []

    def add_callback(self, callback: BaseCallbacks) -> None:
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def add_callbacks(
        self, callbacks_: Union[Sequence[BaseCallbacks], "CallbackManager"]
    ) -> None:
        if isinstance(callbacks_, CallbackManager):
            self.callbacks.extend(callbacks_.callbacks)
        elif isinstance(callbacks_, Sequence):
            for callback in callbacks_:
                if isinstance(callback, BaseCallbacks):
                    self.callbacks.append(callback)

    def remove_callback(self, callback: BaseCallbacks) -> None:
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def __contains__(self, callback: BaseCallbacks) -> bool:
        return callback in self.callbacks

    def __len__(self) -> int:
        return len(self.callbacks)

    def __getitem__(self, index: int) -> BaseCallbacks:
        return self.callbacks[index]

    def __setitem__(self, index: int, callback: BaseCallbacks) -> None:
        self.callbacks[index] = callback

    def __delitem__(self, index: int) -> None:
        del self.callbacks[index]

    def __iter__(self):
        return iter(self.callbacks)

    def on_availability_check(self, available: bool) -> None:
        for callback in self.callbacks:
            callback.on_availability_check(available)

    def on_model_list(self, models: List[str], **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_model_list(models, **kwargs)

    def on_best_model_selected(self, model_name: str, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_best_model_selected(model_name, **kwargs)

    def on_text_generation(
        self, messages: List[Message], response: Message, **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_text_generation(messages, response, **kwargs)

    def on_text_streaming(self, fragment: str, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_text_streaming(fragment, **kwargs)

    def on_audio_transcription(
        self, transcription: str, audio_uri: str, **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_audio_transcription(transcription, audio_uri, **kwargs)

    def on_document_conversion(
        self, document_uri: str, markdown: str, **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_document_conversion(document_uri, markdown, **kwargs)

    def on_video_conversion(
        self, video_uri: str, images: List[str], audio: str, **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_video_conversion(video_uri, images, audio, **kwargs)

    def on_audio_generation(self, text: str, audio_uri: str, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_audio_generation(text, audio_uri, **kwargs)

    def on_image_generation(self, prompt: str, image: Any, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_image_generation(prompt, image, **kwargs)

    def on_text_embedding(
        self, texts: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_text_embedding(texts, embeddings, **kwargs)

    def on_image_embedding(
        self, image_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_image_embedding(image_uris, embeddings, **kwargs)

    def on_audio_embedding(
        self, audio_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_audio_embedding(audio_uris, embeddings, **kwargs)

    def on_video_embedding(
        self, video_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_video_embedding(video_uris, embeddings, **kwargs)

    def on_document_embedding(
        self, document_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_document_embedding(document_uris, embeddings, **kwargs)

    def on_image_description(self, image_uri: str, description: str, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_image_description(image_uri, description, **kwargs)

    def on_audio_description(self, audio_uri: str, description: str, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_audio_description(audio_uri, description, **kwargs)

    def on_video_description(self, video_uri: str, description: str, **kwargs) -> None:
        for callback in self.callbacks:
            callback.on_video_description(video_uri, description, **kwargs)

    def on_document_description(
        self, video_uri: str, description: str, **kwargs
    ) -> None:
        for callback in self.callbacks:
            callback.on_document_description(video_uri, description, **kwargs)
