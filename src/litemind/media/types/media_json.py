from typing import Optional, Union

from litemind.media.media_default import MediaDefault
from litemind.media.types.media_text import Text


class Json(MediaDefault):
    """
    A media that stores JSON
    """

    def __init__(self, json: Optional[Union[str, dict]], **kwargs):
        """
        Create a new JSON media.

        Parameters
        ----------
        json: Optional[Union[str, dict]]
            The JSON content as a string or a dictionary.

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
        """
        Create a JSON media from a string.
        """
        return cls(json=param)

    def get_content(self) -> dict:
        return self.json

    def to_markdown_string(self):
        """
        Convert the text to a markdown representation.
        """
        # Convert JSON to markdown:
        markdown = f"```json\n{str(self)}\n```"

        return markdown

    def to_markdown_text_media(self):
        """
        Convert the text to a markdown representation.
        """
        return Text(self.to_markdown_string())

    def __str__(self):
        import json as js

        # Dump json as a string:
        json_str = js.dumps(self.json)

        return json_str

    def __repr__(self):
        return str(self)

    def __len__(self) -> int:
        return len(str(self.json))
