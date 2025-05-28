from pprint import pformat
from typing import Any, List, Sequence

from arbol import aprint

from litemind.agent.messages.message import Message
from litemind.apis.callbacks.base_callbacks import BaseCallbacks


class PrintCallbacks(BaseCallbacks):

    def __init__(
        self,
        print_model_list: bool = False,
        print_best_model_selected: bool = False,
        print_text_generation: bool = True,
        print_text_streaming: bool = False,
        print_audio_transcription: bool = False,
        print_document_conversion: bool = False,
        print_video_conversion: bool = False,
        print_audio_generation: bool = True,
        print_image_generation: bool = True,
        print_text_embedding: bool = False,
        print_image_embedding: bool = False,
        print_audio_embedding: bool = False,
        print_video_embedding: bool = False,
        print_document_embedding: bool = False,
        print_image_description: bool = False,
        print_audio_description: bool = False,
        print_video_description: bool = False,
        print_document_description: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.print_model_list = print_model_list
        self.print_best_model_selected = print_best_model_selected
        self.print_text_generation = print_text_generation
        self.print_text_streaming = print_text_streaming
        self.print_audio_transcription = print_audio_transcription
        self.print_document_conversion = print_document_conversion
        self.print_video_conversion = print_video_conversion
        self.print_audio_generation = print_audio_generation
        self.print_image_generation = print_image_generation
        self.print_text_embedding = print_text_embedding
        self.print_image_embedding = print_image_embedding
        self.print_audio_embedding = print_audio_embedding
        self.print_video_embedding = print_video_embedding
        self.print_document_embedding = print_document_embedding
        self.print_image_description = print_image_description
        self.print_audio_description = print_audio_description
        self.print_video_description = print_video_description
        self.print_document_description = print_document_description

        if kwargs:
            raise ValueError(f"Unknown arguments: {pformat(kwargs)}")

    def on_availability_check(self, available: bool) -> None:
        aprint(f"Availability Check: {available}")

    def on_model_list(self, models: List[str], **kwargs) -> None:
        if self.print_model_list:
            aprint(f"Model List: {models}")
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_best_model_selected(self, model_name: str, **kwargs) -> None:
        if self.print_best_model_selected:
            aprint(f"Best Model Selected: {model_name}")
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_text_generation(
        self, messages: List[Message], response: Message, **kwargs
    ) -> None:
        if self.print_text_generation:
            aprint(f"Text Generation: Messages: {messages}, Response: {response}")
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_text_streaming(self, fragment: str, **kwargs) -> None:
        if self.print_text_streaming:
            aprint(f"Text Streaming: {fragment}")
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_audio_transcription(
        self, transcription: str, audio_uri: str, **kwargs
    ) -> None:
        if self.print_audio_transcription:
            aprint(f"Audio Transcription: {transcription}, Audio URI: {audio_uri}")
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_document_conversion(
        self, document_uri: str, markdown: str, **kwargs
    ) -> None:
        if self.print_document_conversion:
            aprint(
                f"Document Conversion: Document URI: {document_uri}, Markdown: {markdown}"
            )
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_video_conversion(
        self, video_uri: str, images: List[str], audio: str, **kwargs
    ) -> None:
        if self.print_video_conversion:
            aprint(
                f"Video Conversion: Video URI: {video_uri}, Images: {images}, Audio: {audio}"
            )
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_audio_generation(self, text: str, audio_uri: str, **kwargs) -> None:
        if self.print_audio_generation:
            aprint(f"Audio Generation: Text: {text}, Audio URI: {audio_uri}")
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_image_generation(self, prompt: str, image: Any, **kwargs) -> None:
        if self.print_image_generation:
            aprint(f"Image Generation: Prompt: {prompt}, Image: {image}")
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_text_embedding(
        self, texts: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        if self.print_text_embedding:
            aprint(f"Text Embedding: Texts: {texts}, Embeddings: {embeddings}")
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_image_embedding(
        self, image_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        if self.print_image_embedding:
            aprint(
                f"Image Embedding: Image URIs: {image_uris}, Embeddings: {embeddings}"
            )
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_audio_embedding(
        self, audio_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        if self.print_audio_embedding:
            aprint(
                f"Audio Embedding: Audio URIs: {audio_uris}, Embeddings: {embeddings}"
            )
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_video_embedding(
        self, video_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        if self.print_video_embedding:
            aprint(
                f"Video Embedding: Video URIs: {video_uris}, Embeddings: {embeddings}"
            )
            if kwargs:
                aprint(f"Additional arguments: {pformat(kwargs)}")

    def on_document_embedding(
        self, document_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        if self.print_document_embedding:
            aprint(
                f"Document Embedding: Document URIs: {document_uris}, Embeddings: {embeddings}"
            )
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

    def on_document_description(
        self, document_uri: str, description: str, **kwargs
    ) -> None:
        aprint(
            f"Document Description: Document URI: {document_uri}, Description: {description}"
        )
        if kwargs:
            aprint(f"Additional arguments: {pformat(kwargs)}")
