from litemind.media.media_default import MediaDefault


class Text(MediaDefault):
    """
    A media that stores plain text
    """

    def __init__(self, text: str, **kwargs):
        """
        Create a new text information.

        Parameters
        ----------
        text: str
            The plain text content.
        kwargs: dict
            Other arguments passed to MediaDefault.

        """

        super().__init__(**kwargs)

        # Check that it is really a string:
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, not {type(text)}")

        # Set attributes:
        self.text = text

    def get_content(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text
