import pytest

from litemind.agent.messages.message import Message
from litemind.apis.callbacks.print_api_callbacks import PrintApiCallbacks


@pytest.fixture
def print_callbacks():
    return PrintApiCallbacks(
        print_text_embedding=True,
        print_audio_embedding=True,
        print_video_embedding=True,
        print_image_embedding=True,
        print_document_embedding=True,
        print_audio_description=True,
        print_image_description=True,
        print_audio_generation=True,
        print_video_description=True,
        print_document_description=True,
        print_image_generation=True,
        print_video_conversion=True,
        print_document_conversion=True,
        print_audio_transcription=True,
        print_text_streaming=True,
        print_text_generation=True,
        print_best_model_selected=True,
        print_model_list=True,
    )


def test_on_availability_check(print_callbacks, capsys):
    print_callbacks.on_availability_check(True)
    captured = capsys.readouterr()
    assert "Availability Check: True" in captured.out


def test_on_model_list(print_callbacks, capsys):
    print_callbacks.on_model_list(["model1", "model2"], param="value")
    captured = capsys.readouterr()
    assert "Model List: ['model1', 'model2']" in captured.out
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_best_model_selected(print_callbacks, capsys):
    print_callbacks.on_best_model_selected("best_model", param="value")
    captured = capsys.readouterr()
    assert "Best Model Selected: best_model" in captured.out
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_text_generation(print_callbacks, capsys):
    messages = [Message(role="user", text="Hello"), Message(role="user", text="World")]
    response = Message("Response")
    print_callbacks.on_text_generation(messages, response, param="value")
    captured = capsys.readouterr()
    assert "Text Generation: Messages: [*user*:" in captured.out
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_text_streaming(print_callbacks, capsys):
    print_callbacks.on_text_streaming("fragment", param="value")
    captured = capsys.readouterr()
    assert "Text Streaming: fragment" in captured.out
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_audio_transcription(print_callbacks, capsys):
    print_callbacks.on_audio_transcription("transcription", "audio_uri", param="value")
    captured = capsys.readouterr()
    assert "Audio Transcription: transcription, Audio URI: audio_uri" in captured.out
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_document_conversion(print_callbacks, capsys):
    print_callbacks.on_document_conversion("document_uri", "markdown", param="value")
    captured = capsys.readouterr()
    assert (
        "Document Conversion: Document URI: document_uri, Markdown: markdown"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_video_conversion(print_callbacks, capsys):
    print_callbacks.on_video_conversion(
        "video_uri", ["image1", "image2"], "audio", param="value"
    )
    captured = capsys.readouterr()
    assert (
        "Video Conversion: Video URI: video_uri, Images: ['image1', 'image2'], Audio: audio"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_audio_generation(print_callbacks, capsys):
    print_callbacks.on_audio_generation("text", "audio_uri", param="value")
    captured = capsys.readouterr()
    assert "Audio Generation: Text: text, Audio URI: audio_uri" in captured.out
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_image_generation(print_callbacks, capsys):
    print_callbacks.on_image_generation("prompt", "image", param="value")
    captured = capsys.readouterr()
    assert "Image Generation: Prompt: prompt, Image: image" in captured.out
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_text_embedding(print_callbacks, capsys):
    print_callbacks.on_text_embedding(
        ["text1", "text2"], [[0.1, 0.2], [0.3, 0.4]], param="value"
    )
    captured = capsys.readouterr()
    assert (
        "Text Embedding: Texts: ['text1', 'text2'], Embeddings: [[0.1, 0.2], [0.3, 0.4]]"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_image_embedding(print_callbacks, capsys):
    print_callbacks.on_image_embedding(
        ["image_uri1", "image_uri2"], [[0.1, 0.2], [0.3, 0.4]], param="value"
    )
    captured = capsys.readouterr()
    assert (
        "Image Embedding: Image URIs: ['image_uri1', 'image_uri2'], Embeddings: [[0.1, 0.2], [0.3, 0.4]]"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_audio_embedding(print_callbacks, capsys):
    print_callbacks.on_audio_embedding(
        ["audio_uri1", "audio_uri2"], [[0.1, 0.2], [0.3, 0.4]], param="value"
    )
    captured = capsys.readouterr()
    assert (
        "Audio Embedding: Audio URIs: ['audio_uri1', 'audio_uri2'], Embeddings: [[0.1, 0.2], [0.3, 0.4]]"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_video_embedding(print_callbacks, capsys):
    print_callbacks.on_video_embedding(
        ["video_uri1", "video_uri2"], [[0.1, 0.2], [0.3, 0.4]], param="value"
    )
    captured = capsys.readouterr()
    assert (
        "Video Embedding: Video URIs: ['video_uri1', 'video_uri2'], Embeddings: [[0.1, 0.2], [0.3, 0.4]]"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_document_embedding(print_callbacks, capsys):
    print_callbacks.on_document_embedding(
        ["document_uri1", "document_uri2"], [[0.1, 0.2], [0.3, 0.4]], param="value"
    )
    captured = capsys.readouterr()
    assert (
        "Document Embedding: Document URIs: ['document_uri1', 'document_uri2'], Embeddings: [[0.1, 0.2], [0.3, 0.4]]"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_image_description(print_callbacks, capsys):
    print_callbacks.on_image_description("image_uri", "description", param="value")
    captured = capsys.readouterr()
    assert (
        "Image Description: Image URI: image_uri, Description: description"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_audio_description(print_callbacks, capsys):
    print_callbacks.on_audio_description("audio_uri", "description", param="value")
    captured = capsys.readouterr()
    assert (
        "Audio Description: Audio URI: audio_uri, Description: description"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_video_description(print_callbacks, capsys):
    print_callbacks.on_video_description("video_uri", "description", param="value")
    captured = capsys.readouterr()
    assert (
        "Video Description: Video URI: video_uri, Description: description"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out


def test_on_document_description(print_callbacks, capsys):
    print_callbacks.on_document_description(
        "document_uri", "description", param="value"
    )
    captured = capsys.readouterr()
    assert (
        "Document Description: Document URI: document_uri, Description: description"
        in captured.out
    )
    assert "Additional arguments: {'param': 'value'}" in captured.out
