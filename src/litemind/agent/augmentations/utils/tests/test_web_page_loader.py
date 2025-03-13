import pytest
from litemind.agent.augmentations.utils.web_page_loader import WebPageLoader
from litemind.agent.augmentations.augmentation_base import Document

def test_load_url_success():
    url = "https://example.com"
    document = WebPageLoader.load_url(url)

    assert isinstance(document, Document)
    assert "Example Domain" in document.content
    assert document.metadata["url"] == url
    assert document.metadata["source"] == "web_page"

def test_load_url_error():
    url = "https://nonexistent.example.com"

    try:
        WebPageLoader.load_url(url)
        assert False
    except Exception as e:
        print (f"Error happened as expected: {e}")
        assert True


if __name__ == "__main__":
    pytest.main()