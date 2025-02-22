from typing import List, Any, Sequence

from arbol import aprint
from pprint import pformat

from litemind.agent.message import Message
from litemind.apis._callbacks.base_callbacks import BaseCallbacks


class PrintCallbacks(BaseCallbacks):
    def on_availability_check(self, available: bool) -> None:
        aprint(f"Availability Check: {available}")

    def on_model_list(self, models: List[str], **kwargs) -> None:
        aprint(f"Model List: {models}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_best_model_selected(self, model_name: str, **kwargs) -> None:
        aprint(f"Best Model Selected: {model_name}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_text_generation(self, messages: List[Message], response: Message, **kwargs) -> None:
        aprint(f"Text Generation: Messages: {messages}, Response: {response}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_text_streaming(self, fragment: str, **kwargs) -> None:
        aprint(f"Text Streaming: {fragment}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_audio_transcription(self, transcription: str, audio_uri: str, **kwargs) -> None:
        aprint(f"Audio Transcription: {transcription}, Audio URI: {audio_uri}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_document_conversion(self, document_uri: str, markdown: str, **kwargs) -> None:
        aprint(f"Document Conversion: Document URI: {document_uri}, Markdown: {markdown}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_video_conversion(self, video_uri: str, images: List[str], audio: str, **kwargs) -> None:
        aprint(f"Video Conversion: Video URI: {video_uri}, Images: {images}, Audio: {audio}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_audio_generation(self, text: str, audio_uri: str, **kwargs) -> None:
        aprint(f"Audio Generation: Text: {text}, Audio URI: {audio_uri}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_image_generation(self, prompt: str, image: Any, **kwargs) -> None:
        aprint(f"Image Generation: Prompt: {prompt}, Image: {image}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_text_embedding(self, texts: List[str], embeddings: Sequence[Sequence[float]], **kwargs) -> None:
        aprint(f"Text Embedding: Texts: {texts}, Embeddings: {embeddings}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_image_embedding(self, image_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs) -> None:
        aprint(f"Image Embedding: Image URIs: {image_uris}, Embeddings: {embeddings}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_audio_embedding(self, audio_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs) -> None:
        aprint(f"Audio Embedding: Audio URIs: {audio_uris}, Embeddings: {embeddings}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_video_embedding(self, video_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs) -> None:
        aprint(f"Video Embedding: Video URIs: {video_uris}, Embeddings: {embeddings}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_image_description(self, image_uri: str, description: str, **kwargs) -> None:
        aprint(f"Image Description: Image URI: {image_uri}, Description: {description}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_audio_description(self, audio_uri: str, description: str, **kwargs) -> None:
        aprint(f"Audio Description: Audio URI: {audio_uri}, Description: {description}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_video_description(self, video_uri: str, description: str, **kwargs) -> None:
        aprint(f"Video Description: Video URI: {video_uri}, Description: {description}")
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")