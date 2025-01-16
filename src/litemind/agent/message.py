from abc import ABC
from typing import Optional, List


class Message(ABC):

    def __init__(self, role: str,
                 text: Optional[str] = None,
                 image_uri: Optional[str] = None,
                 audio_uri: Optional[str] = None,
                 video_uri: Optional[str] = None):
        """
        Create a new message.

        Parameters
        ----------
        role: str
            The role of the message (e.g., 'user' or 'agent').
        text: Optional[str]
            The text of the message.
        image_uri: Optional[str]
            The URI of the image to include in the message.
        audio_uri: Optional[str]
            The URI of the audio to include in the message.
        video_uri: Optional[str]
            The URI of the video to include in the message
        """

        self.role = role
        self.text = "" if text is None else text
        self.image_uris = [] if image_uri is None else [image_uri]
        self.audio_uris = [] if audio_uri is None else [audio_uri]
        self.video_uris = [] if video_uri is None else [video_uri]

    def copy(self):
        """
        Create a copy of the message.

        Returns
        -------
        Message
            A copy of the message.
        """
        return Message(role=self.role,
                       text=self.text,
                       image_uri=self.image_uris[
                           0] if self.image_uris else None,
                       audio_uri=self.audio_uris[
                           0] if self.audio_uris else None,
                       video_uri=self.video_uris[
                           0] if self.video_uris else None)

    def append_text(self, text: str):
        """
        Append text to the message.

        Parameters
        ----------
        text : str
            The text to append to the message.

        """
        self.text += text

    def _append_uri(self, uri: str, uri_list: List[str],
                    valid_prefixes: List[str]):
        """
        Append a URI to the specified list if it has a valid prefix.

        Parameters
        ----------
        uri : str
            The URI to append.
        uri_list : List[str]
            The list to append the URI to.
        valid_prefixes : List[str]
            The list of valid prefixes for the URI.
        """
        if not any(uri.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid URI: '{uri}' (must start with one of {valid_prefixes})")
        uri_list.append(uri)

    def _append_url(self, url: str, uri_list: List[str]):
        """
        Append a URL to the specified list if it starts with 'http://' or 'https://'.

        Parameters
        ----------
        url : str
            The URL to append.
        uri_list : List[str]
            The list to append the URL to.
        """
        self._append_uri(url, uri_list, ["http://", "https://"])

    def _append_path(self, path: str, uri_list: List[str]):
        """
        Append a file path to the specified list if it starts with 'file://'.

        Parameters
        ----------
        path : str
            The file path to append.
        uri_list : List[str]
            The list to append the file path to.
        """
        self._append_uri(path, uri_list, ["file://"])

    def append_image_uri(self, image_uri: str):
        self._append_uri(image_uri, self.image_uris,
                         ["data:image/", "http://", "https://", "file://"])

    def append_image_url(self, image_url: str):
        self._append_url(image_url, self.image_uris)

    def append_image_path(self, image_path: str):
        self._append_path(image_path, self.image_uris)

    def append_audio_uri(self, audio_uri: str):
        self._append_uri(audio_uri, self.audio_uris,
                         ["data:audio/", "http://", "https://", "file://"])

    def append_audio_url(self, audio_url: str):
        self._append_url(audio_url, self.audio_uris)

    def append_audio_path(self, audio_path: str):
        self._append_path(audio_path, self.audio_uris)

    def append_video_uri(self, video_uri: str):
        self._append_uri(video_uri, self.video_uris,
                         ["data:video/", "http://", "https://", "file://"])

    def append_video_url(self, video_url: str):
        self._append_url(video_url, self.video_uris)

    def append_video_path(self, video_path: str):
        self._append_path(video_path, self.video_uris)

    def __contains__(self, item: str) -> bool:
        """
        Check if the given string is in the message text, image URLs, or audio URLs.

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
        for audio_uri in self.audio_uris:
            if item in audio_uri:
                return True
        for video_uri in self.video_uris:
            if item in video_uri:
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
        message_string += '\n'
        for image_uri in self.image_uris:
            message_string += f"Image: {image_uri}\n"
        for audio_uri in self.audio_uris:
            message_string += f"Audio: {audio_uri}\n"
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
