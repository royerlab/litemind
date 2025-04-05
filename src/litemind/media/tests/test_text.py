from litemind.media.types.media_text import Text


class TestText:
    def test_init(self):
        """Test initializing Text with valid parameters."""
        text = Text(text="Hello, world!")
        assert text.text == "Hello, world!"

    def test_str(self):
        """Test the string representation of Text."""
        text = Text(text="Hello, world!")
        assert str(text) == "Hello, world!"

    def test_len(self):
        """Test length calculation for Text."""
        text = Text(text="Hello, world!")
        assert len(text) == 13

    def test_empty_text(self):
        """Test initializing Text with empty string."""
        text = Text(text="")
        assert text.text == ""
        assert len(text) == 0
