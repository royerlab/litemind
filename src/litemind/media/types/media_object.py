from typing import Any

from pydantic import BaseModel

from litemind.media.media_default import MediaDefault
from litemind.media.types.media_json import Json
from litemind.media.types.media_text import Text


class Object(MediaDefault):
    """Media that wraps a Pydantic BaseModel instance.

    The object is serialised to JSON for display, hashing, and conversion
    purposes via Pydantic's ``model_dump_json``.
    """

    def __init__(self, object_: BaseModel, **kwargs):
        """Create a new object media.

        Parameters
        ----------
        object_ : BaseModel
            A Pydantic model instance to wrap.
        **kwargs
            Additional keyword arguments forwarded to ``MediaDefault``.

        Raises
        ------
        ValueError
            If *object_* is None or not a Pydantic BaseModel.
        """

        super().__init__(**kwargs)

        # Validate Object here:
        if object_ is None:
            raise ValueError("Object cannot be None")
        elif not isinstance(object_, BaseModel):
            raise ValueError(
                f"Object must be a Pydantic BaseModel, got {type(object_)}"
            )

        # Set attributes:
        self.object = object_

    def get_content(self) -> Any:
        """Return the wrapped Pydantic model instance.

        Returns
        -------
        BaseModel
            The stored Pydantic object.
        """
        return self.object

    def to_json_string(self) -> str:
        """Serialize the object to a JSON string.

        Returns
        -------
        str
            The JSON string produced by ``model_dump_json()``.
        """

        # Convert the object to a JSON string
        return self.object.model_dump_json()

    def to_json_media(self) -> Json:
        """Convert the object to a Json media.

        Returns
        -------
        Json
            A Json media wrapping the serialised object.
        """

        # Create a new Json instance
        return Json(self.to_json_string())

    def to_markdown_string(self) -> str:
        """Render the object as a Markdown JSON fenced code block.

        Returns
        -------
        str
            A Markdown string with the JSON wrapped in triple backticks.
        """

        # Convert object to json since it is a pydantic object:
        json_str = self.to_json_string()

        # Convert JSON string to markdown:
        markdown = f"```json\n{json_str}\n```"

        return markdown

    def to_markdown_text_media(self) -> Text:
        """Convert the object to a Text media with Markdown formatting.

        Returns
        -------
        Text
            A Text media containing the Markdown representation.
        """

        # Convert object to markdown string:
        return Text(self.to_markdown_string())

    def __str__(self) -> str:
        return self.to_json_string()

    def __len__(self) -> int:
        return len(self.to_json_string())

    def __hash__(self) -> int:
        return hash(self.to_json_string())
