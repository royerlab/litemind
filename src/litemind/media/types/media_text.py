"""Plain text media type for litemind.

Provides the ``Text`` class which wraps a simple string as a media object,
serving as the most fundamental media type and the common target for most
media converters.
"""

from litemind.media.media_default import MediaDefault


class Text(MediaDefault):
    """Media that stores plain text content.

    This is the most fundamental media type and serves as the common target
    for most media converters.
    """

    def __init__(self, text: str, **kwargs):
        """Create a new text media.

        Parameters
        ----------
        text : str
            The plain text content.
        **kwargs
            Additional keyword arguments forwarded to ``MediaDefault``.

        Raises
        ------
        ValueError
            If *text* is not a string.
        """

        super().__init__(**kwargs)

        # Check that it is really a string:
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, not {type(text)}")

        # Set attributes:
        self.text = text

    def get_content(self) -> str:
        """Return the plain text content.

        Returns
        -------
        str
            The stored text string.
        """
        return self.text

    def __str__(self) -> str:
        return self.text
