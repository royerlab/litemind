from typing import Any

from pydantic import BaseModel

from litemind.media.media_default import MediaDefault
from litemind.media.types.media_json import Json
from litemind.media.types.media_text import Text


class Object(MediaDefault):
    """
    A media that stores objects that derive from baseModel (Pydantic)
    """

    def __init__(self, object_: BaseModel, **kwargs):
        """
        Create a new object information.

        Parameters
        ----------
        object_: BaseModel
            Object to store

        """

        super().__init__(**kwargs)

        # Validate Object here:
        if object_ is None:
            raise ValueError(f"Object cannot be None")
        elif not isinstance(object_, BaseModel):
            raise ValueError(
                f"Object must be a Pydantic BaseModel, got {type(object_)}"
            )

        # Set attributes:
        self.object = object_

    def get_content(self) -> Any:
        return self.object

    def to_json_string(self) -> str:
        """
        Convert the object to a JSON string.

        Returns
        -------
        str
            The JSON string representation of the object.
        """

        # Convert the object to a JSON string
        return self.object.model_dump_json()

    def to_json_media(self) -> Json:
        """
        Convert the object to a JSON information.

        Returns
        -------
        InformationJson
            The JSON information representation of the object.
        """

        # Create a new InformationJson instance
        return Json(self.to_json_string())

    def to_markdown_string(self) -> str:

        # Convert object to json since it is a pydantic object:
        json_str = self.to_json_string()

        # Convert JSON string to markdown:
        markdown = f"```json\n{json_str}\n```"

        return markdown

    def to_markdown_text_media(self) -> Text:
        """
        Convert the object to a markdown representation as a Text media.

        Returns
        -------
        str
            The markdown string representation of the object.
        """

        # Convert object to markdown string:
        return Text(self.to_markdown_string())

    def __str__(self) -> str:
        return self.to_json_string()

    def __len__(self) -> int:
        return len(self.to_json_string())
