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
            The url of the image to append to the message.

        """
        self.image_uris.append(image_uri)

    def append_image_path(self, image_path: str):
        """
        Append image path to the message.

        Parameters
        ----------
        image_path : str
            The path of the image to append to the message.

        """
        if image_path.endswith('.png'):
            image_format = 'png'
        elif image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
            image_format = 'jpeg'
        else:
            raise NotImplementedError(
                f"Image format not supported: '{image_path}' (only .png and .jpg are supported)")

        import base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            self.append_image_uri(
                f"data:image/{image_format};base64,{encoded_image}")

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
