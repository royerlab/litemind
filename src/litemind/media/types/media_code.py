from litemind.media.media_default import MediaDefault
from litemind.media.types.media_text import Text


class Code(MediaDefault):
    """
    A media that stores computer code in a given language
    """

    def __init__(self, code: str, lang: str):
        """
        Create a new code information.

        Parameters
        ----------
        code: str
            The source code string.
        lang: str
            The programming language of the source code.

        """

        # Check that it is really a string:
        if not isinstance(code, str):
            raise ValueError(f"Parameter 'code' must be a string, not {type(code)}")

        # Set code attribute:
        self.code = code

        # Check that it is really a string:
        if not isinstance(lang, str):
            raise ValueError(f"Parameter 'lang' must be a string, not {type(code)}")

        # Set language attribute:
        self.lang = lang.lower()

    def get_content(self) -> str:
        return self.code

    def to_markdown(self):

        # Convert code to markdown:
        markdown = f"```{self.lang}\n{self.code}\n```"

        return markdown

    def to_markdown_text_media(self):

        # Get Text media:
        text = Text(self.to_markdown())

        return text

    def __str__(self) -> str:
        return self.code
