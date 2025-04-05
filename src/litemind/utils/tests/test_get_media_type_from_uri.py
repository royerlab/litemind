import pytest

from litemind.utils.get_media_type_from_uri import get_media_type_from_uri


@pytest.mark.parametrize(
    "uri, expected_media_type",
    [
        ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA", "image/png"),
        ("http://example.com/image.jpg", "image/jpeg"),
        ("https://example.com/video.mp4", "video/mp4"),
        ("file:///path/to/audio.mp3", "audio/mpeg"),
        ("http://example.com/document.pdf", "application/pdf"),
        ("https://example.com/file.docx", "application/msword"),
        ("file:///path/to/spreadsheet.xlsx", "application/vnd.ms-excel"),
        ("http://example.com/presentation.pptx", "application/vnd.ms-powerpoint"),
        ("https://example.com/text.txt", "text/plain"),
        ("file:///path/to/webpage.html", "text/html"),
        ("http://example.com/data.json", "application/json"),
        ("https://example.com/config.xml", "application/xml"),
        ("file:///path/to/book.epub", "application/epub+zip"),
        ("http://example.com/data.csv", "text/csv"),
        ("https://example.com/data.tab", "text/tab-separated-values"),
        ("http://example.com/unknown.xyz", None),
    ],
)
def test_get_media_type_from_uri(uri, expected_media_type):
    assert get_media_type_from_uri(uri) == expected_media_type
