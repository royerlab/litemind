from typing import List, Type

from arbol import aprint

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase


class MessageConverter:
    """
    This class implements the conversion of a list messages containing arbitrary media to a list of messages containing
    only certain allowed media. Media not covered are converted to an allowed media.

    Media converters can be added to the list of media converters. The conversion is done by using the media converters defined in the class.
    The first media converter in the list able to convert a media takes precedence.
    """

    def __init__(self):

        # List of media converters
        self.media_converters: List[BaseConverter] = []

    def add_default_converters(self) -> List[BaseConverter]:
        """
        Add default converters to convert all media to text.

        Returns
        -------
        List[BaseConverter]
            The list of default converters.

        """

        # Create a list of default converters
        default_converters = []

        # Import the default converters
        from litemind.media.conversion.converters.object_converter import ObjectConverter
        from litemind.media.conversion.converters.table_converter import TableConverter
        from litemind.media.conversion.converters.code_converter import CodeConverter
        from litemind.media.conversion.converters.document_converter_txt import DocumentConverterTxt
        from litemind.media.conversion.converters.json_converter import JsonConverter
        from litemind.media.conversion.converters.ndimage_converter import NdImageConverter
        from litemind.media.conversion.converters.file_converter import FileConverter

        # Import the is_* functions:
        from litemind.utils.ffmpeg_utils import is_ffmpeg_available
        from litemind.utils.whisper_transcribe_audio import is_local_whisper_available
        from litemind.media.conversion.converters.document_converter_docling import is_docling_available
        from litemind.media.conversion.converters.document_converter_pymupdf import is_pymupdf_available

        # Add default converters to the list
        default_converters.append(NdImageConverter())
        default_converters.append(TableConverter())
        default_converters.append(ObjectConverter())
        default_converters.append(CodeConverter())
        default_converters.append(JsonConverter())

        if is_ffmpeg_available():
            from litemind.media.conversion.converters.video_converter_ffmpeg import VideoConverterFfmpeg
            # Add the video converter to the list of default converters
            # This converter is only available if ffmpeg is installed
            default_converters.append(VideoConverterFfmpeg())

        if is_pymupdf_available():
            from litemind.media.conversion.converters.document_converter_pymupdf import DocumentConverterPymupdf
            # Add the document converter to the list of default converters
            # This converter is only available if pymupdf is installed
            default_converters.append(DocumentConverterPymupdf())

        if is_docling_available():
            from litemind.media.conversion.converters.document_converter_docling import DocumentConverterDocling
            # Add the document converter to the list of default converters
            # This converter is only available if docling is installed
            default_converters.append(DocumentConverterDocling())

        if is_local_whisper_available():
            from litemind.media.conversion.converters.audio_converter_whisper_local import AudioConverterWhisperLocal
            # Add the audio converter to the list of default converters
            # This converter is only available if local whisper is installed
            default_converters.append(AudioConverterWhisperLocal())

        # This converter is at the end of the list to have the lowest priority
        default_converters.append(DocumentConverterTxt())

        # Add FileConverter to the list of default converters
        default_converters.append(FileConverter())

        # Add default converters to the list of media converters
        self.media_converters.extend(default_converters)

        return default_converters



    def add_media_converter(self, media_converter: BaseConverter):
        """
        Add a media converter to the list of media converters.
        The media converter is a callable that takes a message and returns a converted message.

        Parameters
        ----------
        media_converter: callable
            The media converter to add. It should be a callable that takes a message and returns a converted message.

        """
        self.media_converters.append(media_converter)

    def remove_media_converter(self, media_converter: BaseConverter):
        """
        Remove a media converter from the list of media converters.

        Parameters
        ----------
        media_converter: callable
            The media converter to remove. It should be a callable that takes a message and returns a converted message.

        """
        self.media_converters.remove(media_converter)

    def convert(self,
                messages: list[Message],
                allowed_media_types: list[Type[MediaBase]],
                recursive: bool = True,
                ) -> list[Message]:

        """
        Convert a list of messages to a list of messages containing only certain allowed media.
        The conversion is done by using the media converters defined in the class.
        The conversion is not done in place: a new list of messages is returned.

        Parameters
        ----------
        messages: list[Message]
            The list of messages to convert.
        allowed_media_types: list[MediaBase]
            The list of allowed media types. The media types are defined in the MediaBase class.
        recursive: bool
            Apply conversion recursively until no more conversion is possible. Default is True.

        Returns
        -------
        list[Message]
            The list of messages containing only certain allowed media.

        """

        # List that will hold the converted messages:
        converted_messages = []

        # By default no conversion happened:
        conversion_happened = False

        # Iterate over the messages:
        for message in messages:

            # Create a new message with the same role as the original message:
            new_message = Message(role=message.role)

            # Iterate over the blocks in the message:
            for block in message.blocks:

                # extract attributes from block:
                attributes = block.attributes

                # get the media from the block:
                media = block.media

                if any(isinstance(media, allowed_type) for allowed_type in allowed_media_types):
                    # If the media type is allowed, keep it as is:
                    new_message.append_block(block)
                else:
                    # Otherwise, attempt to convert it using the available converters
                    converted = None

                    # Iterate over the media converters:
                    for converter in self.media_converters:
                        try:
                            # Check if the converter can convert the media:
                            if converter.can_convert(media):
                                # If the converter can convert the media, convert it:
                                converted = converter.convert(media)
                                # Break after the first successful conversion
                                break
                        except Exception as e:
                            aprint("Error during conversion:", e)
                            new_message.append_text(f"Could not convert {type(media)} to {type(allowed_media_types[0])} because of error {e}.")
                            import traceback
                            traceback.print_exc()

                    if converted is not None:
                        conversion_happened = True
                        # If a converter was found, append the converted media
                        for media in converted:
                            new_message.append_block(MessageBlock(media=media, attributes=attributes))
                    else:
                        # If no converter is found, keep the original media
                        aprint("Warning: No converter found for media type:", type(media))
                        new_message.append_block(block)
            converted_messages.append(new_message)

        # If conversion happened, call convert recursively
        if recursive and conversion_happened:
            return self.convert(converted_messages, allowed_media_types, recursive=True)


        return converted_messages