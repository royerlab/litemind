from abc import ABC
from typing import Optional


class Message(ABC):

    def __init__(self, role: str,
                 text: Optional[str] = None,
                 image_url: Optional[str] = None):
        self.role = role
        self.text = "" if text is None else text
        self.image_urls = [] if image_url is None else [image_url]


    def append_text(self, text: str):
        """
        Append text to the message.

        Parameters
        ----------
        text : str
            The text to append to the message.

        """

        # Append content:
        self.text += text


    def append_image_url(self,
                         image_url: str):
        """
        Append image url to the message.

        Parameters
        ----------

        image_url : str
            The url of the image to append to the message.

        """

        # Append content to list:
        self.image_urls.append(image_url)


    def append_image_path(self,
                          image_path: str):
        """
        Append image path to the message.

        Parameters
        ----------

        image_path : str
            The path of the image to append to the message.

        """

        # Check if the image format is supported:
        if image_path.endswith('.png'):
            image_format = 'png'
        elif image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
            image_format = 'jpeg'
        else:
            raise NotImplementedError(f"Image format not supported: '{image_path}' (only .png and .jpg are supported)")

        import base64

        # Read and encode the image in base64
        with open(image_path, "rb") as image_file:

            # Encode the image in base64:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

            # Append the image to the current message:
            self.append_image_url(f"data:image/{image_format};base64,{encoded_image}")


    # method definition so that we can use the 'in' operator:

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
        for image_url in self.image_urls:
            if item in image_url:
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
        # Return the message as a string, add:
        message_string = f"*{self.role}*:\n"
        message_string += self.text
        for image_url in self.image_urls:
            message_string += f"Image: {image_url}\n"

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


