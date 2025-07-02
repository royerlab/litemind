from typing import Any, List, Sequence

import pytest

from litemind import API_IMPLEMENTATIONS
from litemind.agent.messages.message import Message
from litemind.apis.base_api import BaseApi
from litemind.apis.callbacks.api_callback_manager import ApiCallbackManager
from litemind.apis.callbacks.base_api_callbacks import BaseApiCallbacks
from litemind.apis.model_features import ModelFeatures
from litemind.ressources.media_resources import MediaResources


class MockCallback(BaseApiCallbacks):
    """
    A mock callback class that records the methods called on it.
    """

    def __init__(self):
        self.called_methods = []
        self.call_parameters_dump = ""

    def on_availability_check(self, available: bool) -> None:
        self.called_methods.append("on_availability_check")
        self.call_parameters_dump += f"available,"

    def on_model_list(self, models: List[str], **kwargs) -> None:
        self.called_methods.append("on_model_list")
        self.call_parameters_dump += f"models=`{models}`,"

    def on_best_model_selected(self, model_name: str, **kwargs) -> None:
        self.called_methods.append("on_best_model_selected")
        self.call_parameters_dump += f"model_name=`{model_name}`,"

    def on_text_generation(self, messages: List[str], response: str, **kwargs) -> None:
        self.called_methods.append("on_text_generation")
        self.call_parameters_dump += (
            f"messages=`{messages}`, response={response}, kwargs={kwargs},"
        )

    def on_text_streaming(self, fragment: str, **kwargs) -> None:
        self.called_methods.append("on_text_streaming")
        self.call_parameters_dump += f"fragment=`{fragment}`, kwargs={kwargs},"

    def on_audio_transcription(
        self, audio_uri: str, transcription: str, **kwargs
    ) -> None:
        self.called_methods.append("on_audio_transcription")
        self.call_parameters_dump += (
            f"audio_uri=`{audio_uri}`, transcription={transcription}, kwargs={kwargs},"
        )

    def on_document_conversion(
        self, document_uri: str, markdown: str, **kwargs
    ) -> None:
        self.called_methods.append("on_document_conversion")
        self.call_parameters_dump += (
            f"document_uri=`{document_uri}`, markdown={markdown},"
        )

    def on_video_conversion(
        self, video_uri: str, images: List[str], audio: str, **kwargs
    ) -> None:
        self.called_methods.append("on_video_conversion")
        self.call_parameters_dump += (
            f"video_uri=`{video_uri}`, images={images}, audio={audio},"
        )

    def on_audio_generation(self, text: str, audio_uri: str, **kwargs) -> None:
        self.called_methods.append("on_audio_generation")
        self.call_parameters_dump += f"text=`{text}`, audio_uri={audio_uri},"

    def on_image_generation(self, prompt: str, image: Any, **kwargs) -> None:
        self.called_methods.append("on_image_generation")
        self.call_parameters_dump += f"prompt=`{prompt}`, image={image},"

    def on_text_embedding(
        self, texts: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        self.called_methods.append("on_text_embedding")
        self.call_parameters_dump += f"texts=`{texts}`, embeddings={embeddings},"

    def on_image_embedding(
        self, image_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        self.called_methods.append("on_image_embedding")
        self.call_parameters_dump += (
            f"image_uris=`{image_uris}`, embeddings={embeddings},"
        )

    def on_audio_embedding(
        self, audio_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        self.called_methods.append("on_audio_embedding")
        self.call_parameters_dump += (
            f"audio_uris=`{audio_uris}`, embeddings={embeddings},"
        )

    def on_video_embedding(
        self, video_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        self.called_methods.append("on_video_embedding")
        self.call_parameters_dump += (
            f"video_uris=`{video_uris}`, embeddings={embeddings},"
        )

    def on_image_description(self, image_uri: str, description: str, **kwargs) -> None:
        self.called_methods.append("on_image_description")
        self.call_parameters_dump += (
            f"image_uri=`{image_uri}`, description={description},"
        )

    def on_audio_description(self, audio_uri: str, description: str, **kwargs) -> None:
        self.called_methods.append("on_audio_description")
        self.call_parameters_dump += (
            f"audio_uri=`{audio_uri}`, description={description},"
        )

    def on_video_description(self, video_uri: str, description: str, **kwargs) -> None:
        self.called_methods.append("on_video_description")
        self.call_parameters_dump += (
            f"video_uri=`{video_uri}`, description={description},"
        )


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_availability_check(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    api_instance.check_availability_and_credentials()
    assert "on_availability_check" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_model_list(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    api_instance.list_models()
    assert "on_model_list" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_best_model_selected(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    api_instance.get_best_model()
    assert "on_best_model_selected" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_audio_transcription(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    audio_transcription_model_name = api_instance.get_best_model(
        ModelFeatures.AudioTranscription
    )

    if audio_transcription_model_name is None:
        pytest.skip(
            f"{api_class.__name__} does not support audio transcription. Skipping audio transcription tests."
        )

    audio_path = MediaResources.get_local_test_audio_uri("harvard.wav")
    api_instance.transcribe_audio(audio_path, model_name=audio_transcription_model_name)
    assert "on_audio_transcription" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_text_generation(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    text_generation_model_name = api_instance.get_best_model(
        [ModelFeatures.TextGeneration]
    )

    if text_generation_model_name is None:
        pytest.skip(
            f"{api_class.__name__} does not support text generation. Skipping text generation tests."
        )

    messages = [
        Message(
            role="system",
            text="You are an omniscient all-knowing being called Ohmm",
        ),
        Message(role="user", text="What is the meaning of life?"),
    ]
    api_instance.generate_text(model_name=text_generation_model_name, messages=messages)
    assert "on_text_generation" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_text_generation_streaming(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    text_generation_model_name = api_instance.get_best_model(
        [ModelFeatures.TextGeneration]
    )
    if text_generation_model_name is None:
        pytest.skip(
            f"{api_class.__name__} does not support text generation. Skipping text generation tests."
        )
    messages = [
        Message(
            role="system",
            text="You are an omniscient all-knowing being called Ohmm",
        ),
        Message(role="user", text="What is the meaning of life?"),
    ]
    response = api_instance.generate_text(
        model_name=text_generation_model_name, messages=messages
    )

    print(str(response))

    assert "on_text_streaming" in mock_callback.called_methods
    assert "fragment=" in mock_callback.call_parameters_dump


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_audio_generation(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    audio_generation_model_name = api_instance.get_best_model(
        ModelFeatures.AudioGeneration
    )
    if audio_generation_model_name:
        api_instance.generate_audio("Who is the president of the USA?")
        assert "on_audio_generation" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_image_generation(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    image_generation_model_name = api_instance.get_best_model(
        ModelFeatures.ImageGeneration
    )

    if image_generation_model_name is None:
        pytest.skip(
            f"{api_class.__name__} does not support image generation. Skipping image generation tests."
        )

    print(f"Image generation model name: {image_generation_model_name}")

    api_instance.generate_image("A cute fluffy cat.")
    assert "on_image_generation" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_text_embedding(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    text_embedding_model_name = api_instance.get_best_model(
        ModelFeatures.TextEmbeddings
    )
    if text_embedding_model_name:
        api_instance.embed_texts(
            texts=["Cat", "Dog"], model_name=text_embedding_model_name
        )
        assert "on_text_embedding" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_image_embedding(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    image_embedding_model_name = api_instance.get_best_model(
        ModelFeatures.ImageEmbeddings
    )
    if image_embedding_model_name:
        image_1_uri = MediaResources.get_local_test_image_uri("cat.jpg")
        image_2_uri = MediaResources.get_local_test_image_uri("panda.jpg")
        api_instance.embed_images(
            image_uris=[image_1_uri, image_2_uri], model_name=image_embedding_model_name
        )
        assert "on_image_embedding" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_audio_embedding(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    audio_embedding_model_name = api_instance.get_best_model(
        ModelFeatures.AudioEmbeddings
    )
    if audio_embedding_model_name:
        audio_1_uri = MediaResources.get_local_test_audio_uri("harvard.wav")
        audio_2_uri = MediaResources.get_local_test_audio_uri("preamble.wav")
        api_instance.embed_audios(
            audio_uris=[audio_1_uri, audio_2_uri], model_name=audio_embedding_model_name
        )
        assert "on_audio_embedding" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_video_embedding(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    video_embedding_model_name = api_instance.get_best_model(
        ModelFeatures.VideoEmbeddings
    )
    if video_embedding_model_name:
        video_1_uri = MediaResources.get_local_test_video_uri("flying.mp4")
        video_2_uri = MediaResources.get_local_test_video_uri("lunar_park.mp4")
        api_instance.embed_videos(
            video_uris=[video_1_uri, video_2_uri], model_name=video_embedding_model_name
        )
        assert "on_video_embedding" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_image_description(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    image_description_model_name = api_instance.get_best_model(
        [ModelFeatures.TextGeneration, ModelFeatures.Image]
    )
    if image_description_model_name:
        image_uri = MediaResources.get_local_test_image_uri("cat.jpg")
        api_instance.describe_image(image_uri, model_name=image_description_model_name)
        assert "on_image_description" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_audio_description(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    audio_description_model_name = api_instance.get_best_model(
        [ModelFeatures.TextGeneration, ModelFeatures.Audio]
    )
    if audio_description_model_name:
        audio_uri = MediaResources.get_local_test_audio_uri("harvard.wav")
        api_instance.describe_audio(audio_uri, model_name=audio_description_model_name)
        assert "on_audio_description" in mock_callback.called_methods


@pytest.mark.parametrize("api_class", API_IMPLEMENTATIONS)
def test_video_description(api_class):
    callback_manager = ApiCallbackManager()
    mock_callback = MockCallback()
    callback_manager.add_callback(mock_callback)

    api_instance: BaseApi = api_class(callback_manager=callback_manager)

    video_description_model_name = api_instance.get_best_model(
        [ModelFeatures.TextGeneration, ModelFeatures.Video]
    )
    if video_description_model_name:
        video_uri = MediaResources.get_local_test_video_uri("flying.mp4")
        api_instance.describe_video(video_uri, model_name=video_description_model_name)
        assert "on_video_description" in mock_callback.called_methods
