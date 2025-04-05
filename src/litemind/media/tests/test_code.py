import pytest

from litemind.agent.messages.message_block import MessageBlock
from litemind.media.types.media_code import Code
from litemind.media.types.media_text import Text


class TestCode:

    @pytest.fixture
    def sample_code(self):
        """Create a sample Code object for testing."""
        return Code(code="print('Hello, World!')", lang="python")

    def test_code_initialization(self):
        """Test basic initialization with valid parameters"""
        code_str = "print('Hello, world!')"
        lang = "python"
        code = Code(code=code_str, lang=lang)

        assert code.code == code_str
        assert code.lang == lang

    def test_init_valid(self):
        """Test initializing Code with valid parameters."""
        code = Code(code="const x = 5;", lang="javascript")
        assert code.code == "const x = 5;"
        assert code.lang == "javascript"

    def test_init_invalid_code(self):
        """Test initialization with non-string code raises ValueError."""
        with pytest.raises(
            ValueError, match="Parameter 'code' must be a string, not <class 'int'>"
        ):
            Code(code=123, lang="python")

    def test_init_invalid_lang(self):
        """Test initialization with non-string lang raises ValueError."""
        with pytest.raises(
            ValueError, match="Parameter 'lang' must be a string, not <class 'str'>"
        ):
            Code(code="print('test')", lang=123)

    def test_to_markdown(self, sample_code):
        """Test the to_markdown method."""
        markdown = sample_code.to_markdown()
        assert markdown == "```python\nprint('Hello, World!')\n```"

    def test_to_markdown_2(self):
        """Test to_markdown method generates correct markdown format"""
        code_str = "function greet() {\n  console.log('Hello');\n}"
        lang = "javascript"
        code = Code(code=code_str, lang=lang)

        expected_markdown = f"```{lang}\n{code_str}\n```"
        assert code.to_markdown() == expected_markdown

    def test_to_markdown_text_media(self, sample_code):
        """Test the to_markdown_text_media method."""
        text_media = sample_code.to_markdown_text_media()
        assert isinstance(text_media, Text)
        assert text_media.text == "```python\nprint('Hello, World!')\n```"

    def test_to_markdown_text_media_2(self):
        """Test to_markdown_text_media returns a Text object with correct content"""
        code_str = "def greet():\n    print('Hello')"
        lang = "python"
        code = Code(code=code_str, lang=lang)

        text_media = code.to_markdown_text_media()

        assert isinstance(text_media, Text)
        assert text_media.text == f"```{lang}\n{code_str}\n```"

    def test_str(self, sample_code):
        """Test the string representation of Code."""
        assert str(sample_code) == "print('Hello, World!')"

    def test_len(self, sample_code):
        """Test the __len__ method inherited from MediaDefault."""
        assert len(sample_code) == len("print('Hello, World!')")

    def test_to_message_block(self, sample_code):
        """Test the to_message_block method inherited from MediaDefault."""
        message_block = sample_code.to_message_block()
        assert isinstance(message_block, MessageBlock)
        assert message_block.media == sample_code

    def test_multiline_code(self):
        """Test with multiline code."""
        multiline = "def hello():\n    print('Hello World')\n\nhello()"
        code = Code(code=multiline, lang="python")
        assert code.code == multiline
        assert code.to_markdown() == f"```python\n{multiline}\n```"

    def test_empty_code(self):
        """Test with empty code string."""
        code = Code(code="", lang="python")
        assert code.code == ""
        assert code.to_markdown() == "```python\n\n```"

    def test_str_representation(self):
        """Test __str__ method returns the code string"""
        code_str = "SELECT * FROM users;"
        code = Code(code=code_str, lang="sql")

        assert str(code) == code_str

    def test_code_with_multiline_content(self):
        """Test code with multiline content is handled correctly"""
        code_str = "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"
        lang = "python"
        code = Code(code=code_str, lang=lang)

        assert code.code == code_str
        assert code.to_markdown() == f"```{lang}\n{code_str}\n```"

    def test_code_with_special_characters(self):
        """Test code with special markdown characters is handled correctly"""
        code_str = "# Comment with *asterisks* and _underscores_\nprint('Hello')"
        lang = "python"
        code = Code(code=code_str, lang=lang)

        markdown = code.to_markdown()
        assert markdown == f"```{lang}\n{code_str}\n```"
        # Special characters should be preserved inside code blocks
        assert "*asterisks*" in markdown
        assert "_underscores_" in markdown
