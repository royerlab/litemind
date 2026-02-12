"""JSON media type for litemind.

Provides the ``Json`` class which wraps structured JSON data (parsed from a
string or supplied as a Python dictionary) as a media object. Supports
conversion to Markdown code blocks and Text media for LLM consumption.
"""

from typing import Optional, Union

from litemind.media.media_default import MediaDefault
from litemind.media.types.media_text import Text


class Json(MediaDefault):
    """Media that stores JSON data.

    Accepts either a JSON string (which is parsed on construction) or a
    Python dictionary.
    """

    def __init__(self, json: Optional[Union[str, dict]], **kwargs):
        """Create a new JSON media.

        Parameters
        ----------
        json : str or dict
            The JSON content. Strings are parsed with ``json.loads``;
            dictionaries are stored directly.
        **kwargs
            Additional keyword arguments forwarded to ``MediaDefault``.

        Raises
        ------
        ValueError
            If *json* is not a string or dictionary, or if the string is
            not valid JSON.
        """

        super().__init__(**kwargs)

        # Check that it is really a string or a dictionary:
        if not isinstance(json, (str, dict)):
            raise ValueError(f"json must be a string or a dictionary, not {type(json)}")

        # if json is a string, check that it is a valid JSON string:
        if isinstance(json, str):
            try:
                import json as js

                # parse json and load:
                self.json = js.loads(json)

            except ValueError as e:
                raise ValueError(f"Invalid JSON string: {e}")
        else:
            self.json = json

    @classmethod
    def from_string(cls, param):
        """Create a Json media by parsing a JSON string.

        Parameters
        ----------
        param : str
            A valid JSON string.

        Returns
        -------
        Json
            A new Json media instance.
        """
        return cls(json=param)

    def get_content(self) -> dict:
        """Return the parsed JSON data as a dictionary.

        Returns
        -------
        dict
            The JSON content.
        """
        return self.json

    def to_markdown_string(self):
        """Render the JSON as a Markdown fenced code block.

        Returns
        -------
        str
            A Markdown string with the JSON wrapped in triple backticks.
        """
        # Convert JSON to markdown:
        markdown = f"```json\n{str(self)}\n```"

        return markdown

    def to_markdown_text_media(self):
        """Convert this JSON to a Text media containing a Markdown code block.

        Returns
        -------
        Text
            A Text media with the Markdown representation.
        """
        return Text(self.to_markdown_string())

    def __str__(self):
        """Return the JSON data serialised as a compact JSON string.

        Returns
        -------
        str
            The JSON content as a string.
        """
        import json as js

        # Dump json as a string:
        json_str = js.dumps(self.json)

        return json_str

    def __repr__(self):
        """Return the JSON data serialised as a compact JSON string.

        Equivalent to ``__str__``.

        Returns
        -------
        str
            The JSON content as a string.
        """
        return str(self)

    def __len__(self) -> int:
        """Return the length of the string representation of the JSON data.

        Returns
        -------
        int
            Number of characters in ``str(self.json)``.
        """
        return len(str(self.json))

    def __hash__(self) -> int:
        """Return a hash of the JSON data using sorted keys for consistency.

        Returns
        -------
        int
            Hash value computed from the deterministic JSON serialisation.
        """
        import json as js

        # Use sorted keys for consistent hashing of dicts
        return hash(js.dumps(self.json, sort_keys=True))
