from typing import Any, List, Sequence

import pytest

from litemind.agent.messages.message import Message
from litemind.apis.callbacks.base_callbacks import BaseCallbacks
from litemind.apis.callbacks.callback_manager import CallbackManager


class MockCallback(BaseCallbacks):
    """
    A mock callback class that records the methods called on it.
    """

    def __init__(self):
        self.called_methods = []

    def on_availability_check(self, available: bool) -> None:
        self.called_methods.append("on_availability_check")

    def on_model_list(self, models: List[str], **kwargs) -> None:
        self.called_methods.append("on_model_list")

    def on_best_model_selected(self, model_name: str, **kwargs) -> None:
        self.called_methods.append("on_best_model_selected")

    def on_text_generation(self, messages: List[str], response: str, **kwargs) -> None:
        self.called_methods.append("on_text_generation")

    def on_text_streaming(self, fragment: str, **kwargs) -> None:
        self.called_methods.append("on_text_streaming")

    def on_audio_transcription(
        self, audio_uri: str, transcription: str, **kwargs
    ) -> None:
        self.called_methods.append("on_audio_transcription")

    def on_document_conversion(
        self, document_uri: str, markdown: str, **kwargs
    ) -> None:
        self.called_methods.append("on_document_conversion")

    def on_video_conversion(
        self, video_uri: str, images: List[str], audio: str, **kwargs
    ) -> None:
        self.called_methods.append("on_video_conversion")

    def on_audio_generation(self, text: str, audio_uri: str, **kwargs) -> None:
        self.called_methods.append("on_audio_generation")

    def on_image_generation(self, prompt: str, image: Any, **kwargs) -> None:
        self.called_methods.append("on_image_generation")

    def on_text_embedding(
        self, texts: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        self.called_methods.append("on_text_embedding")

    def on_image_embedding(
        self, image_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        self.called_methods.append("on_image_embedding")

    def on_audio_embedding(
        self, audio_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        self.called_methods.append("on_audio_embedding")

    def on_video_embedding(
        self, video_uris: List[str], embeddings: Sequence[Sequence[float]], **kwargs
    ) -> None:
        self.called_methods.append("on_video_embedding")

    def on_image_description(self, image_uri: str, description: str, **kwargs) -> None:
        self.called_methods.append("on_image_description")

    def on_audio_description(self, audio_uri: str, description: str, **kwargs) -> None:
        self.called_methods.append("on_audio_description")

    def on_video_description(self, video_uri: str, description: str, **kwargs) -> None:
        self.called_methods.append("on_video_description")


@pytest.fixture
def callback_manager():
    """
    Fixture for creating a CallbackManager instance.
    Returns
    -------
    CallbackManager
        An instance of the CallbackManager class.

    """
    return CallbackManager()


@pytest.fixture
def mock_callback():
    """
    Fixture for creating a MockCallback instance.
    Returns
    -------
    MockCallback
        An instance of the MockCallback class.

    """
    return MockCallback()


def test_add_and_remove_callback(callback_manager, mock_callback):
    """
    Test adding and removing a callback from the CallbackManager.
    """

    # Add the callback:
    callback_manager.add_callback(mock_callback)

    # Check that the callback is in the manager:
    assert len(callback_manager) == 1
    assert mock_callback in callback_manager

    # Remove the callback:
    callback_manager.remove_callback(mock_callback)

    # Check that the callback is no longer in the manager:
    assert len(callback_manager) == 0
    assert mock_callback not in callback_manager


def test_on_availability_check(callback_manager, mock_callback):
    """
    Test the on_availability_check method of the CallbackManager.
    """

    # Add the callback:
    callback_manager.add_callback(mock_callback)

    # Call the method:
    callback_manager.on_availability_check(True)

    # Check that the callback was called:
    assert "on_availability_check" in mock_callback.called_methods


def test_on_model_list(callback_manager, mock_callback):
    """
    Test the on_model_list method of the CallbackManager.
    """

    # Add the callback:
    callback_manager.add_callback(mock_callback)

    # Call the method:
    callback_manager.on_model_list(["model1", "model2"])

    # Check that the callback was called:
    assert "on_model_list" in mock_callback.called_methods


def test_on_best_model_selected(callback_manager, mock_callback):
    """
    Test the on_best_model_selected method of the CallbackManager.
    """

    # Add the callback:
    callback_manager.add_callback(mock_callback)

    # Call the method:
    callback_manager.on_best_model_selected("best_model")

    # Check that the callback was called:
    assert "on_best_model_selected" in mock_callback.called_methods


def test_on_text_generation(callback_manager, mock_callback):
    """
    Test the on_text_generation method of the CallbackManager.
    """

    # Add the callback:
    callback_manager.add_callback(mock_callback)

    # Make a simple message:
    message = Message(role="user", text="Hello, world!")

    # Make a simple response:
    response = Message(role="assistant", text="Hello you!")

    callback_manager.on_text_generation([message], response)
    assert "on_text_generation" in mock_callback.called_methods


def test_on_text_streaming(callback_manager, mock_callback):
    """
    Test the on_text_streaming method of the CallbackManager.
    """

    # Add the callback:
    callback_manager.add_callback(mock_callback)

    # Call the method:
    callback_manager.on_text_streaming("Hello, world!")

    # Check that the callback was called:
    assert "on_text_streaming" in mock_callback.called_methods
