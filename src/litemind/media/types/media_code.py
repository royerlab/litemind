from litemind.media.media_default import MediaDefault
from litemind.media.types.media_text import Text


class Code(MediaDefault):
    """Media that stores source code in a specific programming language."""

    def __init__(self, code: str, lang: str, **kwargs):
        """Create a new code media.

        Parameters
        ----------
        code : str
            The source code string.
        lang : str
            The programming language identifier (e.g. ``"python"``,
            ``"javascript"``). Stored in lowercase.
        **kwargs
            Additional keyword arguments forwarded to ``MediaDefault``.

        Raises
        ------
        ValueError
            If *code* or *lang* is not a string.
        """

        # Check that it is really a string:
        if not isinstance(code, str):
            raise ValueError(f"Parameter 'code' must be a string, not {type(code)}")

        # Check that it is really a string:
        if not isinstance(lang, str):
            raise ValueError(f"Parameter 'lang' must be a string, not {type(lang)}")

        # Call parent constructor:
        super().__init__(**kwargs)

        # Set code attribute:
        self.code = code

        # Set language attribute:
        self.lang = lang.lower()

    def get_content(self) -> str:
        """Return the source code string.

        Returns
        -------
        str
            The raw source code.
        """
        return self.code

    def to_markdown(self):
        """Render the code as a Markdown fenced code block.

        Returns
        -------
        str
            A Markdown string with the code wrapped in triple backticks
            and annotated with the language identifier.
        """

        # Convert code to markdown:
        markdown = f"```{self.lang}\n{self.code}\n```"

        return markdown

    def to_markdown_text_media(self):
        """Convert this code to a Text media containing a Markdown code block.

        Returns
        -------
        Text
            A Text media whose content is the Markdown representation of
            this code.
        """

        # Get Text media:
        text = Text(self.to_markdown())

        return text

    def __str__(self) -> str:
        return self.code
