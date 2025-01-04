from abc import ABC
from typing import Optional


class Message(ABC):

    def __init__(self, role: str,
                 text: Optional[str] = None,
                 image_uri: Optional[str] = None):
        self.role = role
        self.text = "" if text is None else text
        self.image_uris = [] if image_uri is None else [image_uri]

    def append_text(self, text: str):
        """
        Append text to the message.

        Parameters
        ----------
        text : str
            The text to append to the message.

        """
        self.text += text

    def append_image_uri(self, image_uri: str):
        """
        Append image url to the message.

        Parameters
        ----------
        image_uri : str
            The uri of the image to append to the message.
            Can be: 'data:image/...', 'http://...', 'https://...', or a local file path.

        """
        # Check if the image URI is valid:
        if not image_uri.startswith("data:image/") and \
                not image_uri.startswith("http://") and \
                not image_uri.startswith("https://") and \
                not image_uri.startswith("file://"):
            raise ValueError(
                f"Invalid image URI: '{image_uri}' (must start with 'data:image/', 'http://', 'https://', or 'file://')")

        # Add the image URI to the list of image URIs:
        self.image_uris.append(image_uri)

    def append_image_url(self, image_url: str):
        """
        Append image url to the message.

        Parameters
        ----------
        image_url : str
            The url of the image to append to the message.

        """
        # Check if the image URL is valid:
        if not image_url.startswith("http://") and not image_url.startswith(
                "https://"):
            raise ValueError(
                f"Invalid image URL: '{image_url}' (must start with 'http://' or 'https://')")

        # Add the image URL to the list of image URIs:
        self.image_uris.append(image_url)

    def append_image_path(self, image_path: str):
        """
        Append image path to the message.

        Parameters
        ----------
        image_path : str
            The path of the image to append to the message.

        """

        # First we check if the image path is valid:
        if not image_path.startswith("file://"):
            raise ValueError(
                f"Invalid image path: '{image_path}' (must start with 'file://')")

        # Add the image path to the list of image URIs:
        self.image_uris.append(image_path)

    def __contains__(self, item: str) -> bool:
        """
        Check if the given string is in the message text or image URLs.

        Parameters
        ----------
        item : str
            The string to check for.

        Returns
        -------
        bool
            True if the string is found, False otherwise.
        """
        if item in self.text:
            return True
        for image_uri in self.image_uris:
            if item in image_uri:
                return True
        return False

    def __str__(self):
        """
        Return the message as a string.
        Returns
        -------
        str
            The message as a string.
        """
        message_string = f"*{self.role}*:\n"
        message_string += self.text
        for image_uri in self.image_uris:
            message_string += f"Image: {image_uri}\n"
        return message_string

    def __repr__(self):
        """
        Return the message as a string.
        Returns
        -------
        str
            The message as a string.
        """
        return str(self)
