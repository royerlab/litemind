from typing import List, Optional, Sequence, Set, Type

from arbol import aprint

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_types import all_media_types


class MediaConverter:
    """
    This class implements the conversion of a list messages containing arbitrary media to a list of messages containing
    only certain allowed media. Media not covered are converted to an allowed media.

    Media converters can be added to the list of media converters. The conversion is done by using the media converters defined in the class.
    The first media converter in the list able to convert a media takes precedence.
    """

    def __init__(self):

        # List of media converters
        self.media_converters: List[BaseConverter] = []

    def add_default_converters(
        self,
        convert_audio: bool = True,
        convert_videos: bool = True,
        convert_documents: bool = True,
    ) -> List[BaseConverter]:
        """
        Add default converters to convert all media to text.

        Parameters
        ----------
        convert_audio: bool
            If True, add the audio converter to the list of default converters.
        convert_videos: bool
            If True, add the video converter to the list of default converters.
        convert_documents: bool
            If True, add the document converter to the list of default converters.

        Returns
        -------
        List[BaseConverter]
            The list of default converters.

        """

        # Create a list of default converters
        default_converters = []

        # Import the default converters
        from litemind.media.conversion.converters.code_converter import CodeConverter
        from litemind.media.conversion.converters.document_converter_docling import (
            is_docling_available,
        )
        from litemind.media.conversion.converters.document_converter_pymupdf import (
            is_pymupdf_available,
        )
        from litemind.media.conversion.converters.document_converter_txt import (
            DocumentConverterTxt,
        )
        from litemind.media.conversion.converters.file_converter import FileConverter
        from litemind.media.conversion.converters.json_converter import JsonConverter
        from litemind.media.conversion.converters.ndimage_converter import (
            NdImageConverter,
        )
        from litemind.media.conversion.converters.object_converter import (
            ObjectConverter,
        )
        from litemind.media.conversion.converters.table_converter import TableConverter

        # Import the is_* functions:
        from litemind.utils.ffmpeg_utils import is_ffmpeg_available
        from litemind.utils.whisper_transcribe_audio import is_local_whisper_available

        # Add default converters to the list
        default_converters.append(NdImageConverter())
        default_converters.append(TableConverter())
        default_converters.append(ObjectConverter())
        default_converters.append(CodeConverter())
        default_converters.append(JsonConverter())

        if convert_videos and is_ffmpeg_available():
            from litemind.media.conversion.converters.video_converter_ffmpeg import (
                VideoConverterFfmpeg,
            )

            # Add the video converter to the list of default converters
            # This converter is only available if ffmpeg is installed
            default_converters.append(VideoConverterFfmpeg())

        if convert_documents and is_pymupdf_available():
            from litemind.media.conversion.converters.document_converter_pymupdf import (
                DocumentConverterPymupdf,
            )

            # Add the document converter to the list of default converters
            # This converter is only available if pymupdf is installed
            default_converters.append(DocumentConverterPymupdf())

        if convert_documents and is_docling_available():
            from litemind.media.conversion.converters.document_converter_docling import (
                DocumentConverterDocling,
            )

            # Add the document converter to the list of default converters
            # This converter is only available if docling is installed
            default_converters.append(DocumentConverterDocling())

        if convert_audio and is_local_whisper_available():
            from litemind.media.conversion.converters.audio_converter_whisper_local import (
                AudioConverterWhisperLocal,
            )

            # Add the audio converter to the list of default converters
            # This converter is only available if local whisper is installed
            default_converters.append(AudioConverterWhisperLocal())

        # This converter is at the end of the list to have the lowest priority
        if convert_documents:
            default_converters.append(DocumentConverterTxt())

        # Add FileConverter to the list of default converters
        default_converters.append(FileConverter())

        # Add default converters to the list of media converters
        self.media_converters.extend(default_converters)

        return default_converters

    def add_media_converter(
        self, media_converter: BaseConverter, highest_priority: bool = False
    ):
        """
        Add a media converter to the list of media converters.
        The media converter is a callable that takes a message and returns a converted message.

        Parameters
        ----------
        media_converter: callable
            The media converter to add. It should be a callable that takes a message and returns a converted message.
        highest_priority: bool
            If True, the media converter is added to the beginning of the list of media converters.
            This means that it will be used first when converting messages. Default is False.

        """
        if highest_priority:
            # Add the media converter to the beginning of the list
            self.media_converters.insert(0, media_converter)
        else:
            # Add the media converter to the end of the list
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

    def can_convert_within(
        self,
        source_media_type: Type[MediaBase],
        allowed_media_types: Set[Type[MediaBase]],
    ) -> bool:
        """
        Determines if a source media type, after all possible conversions,
        results in media types that are all contained within the allowed set.

        Parameters
        ----------
        source_media_type: Type[MediaBase]
            The source media type to convert from.
        allowed_media_types: Set[Type[MediaBase]]
            The set of allowed media types that conversions should be limited to.

        Returns
        -------
        bool
            True if all resulting conversions are within the allowed types, False otherwise.
        """

        # If there are no converters, then we cannot convert anything, source type must be in allowed types:
        if not self.media_converters:
            return source_media_type in allowed_media_types

        # Build a conversion graph from all converter rules
        conversion_graph = {}

        # Process all converter rules to build graph
        for converter in self.media_converters:
            for src_type, target_types in converter.rule():
                if src_type not in conversion_graph:
                    conversion_graph[src_type] = set()
                # Add all possible conversion targets
                conversion_graph[src_type].update(target_types)

        # Queue for BFS traversal
        queue = [source_media_type]
        # Keep track of visited types to avoid cycles
        visited = set()

        # BFS to find all possible conversion results
        while queue:
            current_type = queue.pop(0)

            # Skip if already visited
            if current_type in visited:
                continue

            visited.add(current_type)

            # Check if current type can be converted
            if current_type in conversion_graph:
                for next_type in conversion_graph[current_type]:
                    # If any next type is not in allowed types, return False
                    if next_type not in allowed_media_types:
                        return False

                    if next_type not in visited:
                        queue.append(next_type)

        # If we get here, all conversions are within allowed types
        return True

    def get_convertible_media_types(
        self, allowed_media_types: Set[Type[MediaBase]]
    ) -> Set[Type[MediaBase]]:
        """
        Get the set of media types that can be converted to the allowed media types.

        Parameters
        ----------
        allowed_media_types: Set[Type[MediaBase]]
            The set of allowed media types that conversions should be limited to.

        Returns
        -------
        Set[Type[MediaBase]]
            The set of media types that can be converted to the allowed media types.
        """
        convertable_media_types = set()

        # iterate through all media types that derive from MediaBase:
        for media_type in all_media_types():

            # Check if the converter can convert this media type:
            if self.can_convert_within(media_type, allowed_media_types):
                # If it can, add it to the set of convertable media types
                convertable_media_types.add(media_type)

        return convertable_media_types

    def convert(
        self,
        messages: List[Message],
        allowed_media_types: Sequence[Type[MediaBase]],
        exclude_extensions: Optional[Sequence[str]] = None,
        recursive: bool = True,
    ) -> List[Message]:
        """
        Convert a list of messages to a list of messages containing only certain allowed media.
        The conversion is done by using the media converters defined in the class.
        The conversion is not done in place: a new list of messages is returned.

        Parameters
        ----------
        messages: Sequence[Message]
            The list of messages to convert.
        allowed_media_types: Sequence[MediaBase]
            The list of allowed media types. The media types are defined in the MediaBase class.
        exclude_extensions: Optional[Sequence[str]]
            The list of file extensions to exclude from the conversion.
            The file extensions are defined only for some media types.
            If excluded then the conversion is skipped and the corresponding media is kept as is.
            If None, all file extensions are allowed. Default is None.
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

                # Attempt to get the extension if defined:
                extension = getattr(media, "extension", None)

                if any(
                    isinstance(media, allowed_type)
                    for allowed_type in allowed_media_types
                ):
                    # If the media type is allowed, keep it as is:
                    new_message.append_block(block)
                elif (
                    extension is not None
                    and exclude_extensions is not None
                    and extension in exclude_extensions
                ):
                    # If the media type is not allowed and the extension is in the exclude list, skip conversion:
                    # aprint("Warning: Skipping media with excluded extension:", extension)
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
                            new_message.append_text(
                                f"Could not convert {type(media)} to {type(allowed_media_types[0])} because of error {e}."
                            )
                            import traceback

                            traceback.print_exc()

                    if converted is not None:
                        conversion_happened = True
                        # If a converter was found, append the converted media
                        for media in converted:
                            new_message.append_block(
                                MessageBlock(media=media, attributes=attributes)
                            )
                    else:
                        # If no converter is found, keep the original media:
                        new_message.append_block(block)

                        # if the original media is not allowed, log a warning:
                        if not any(
                            isinstance(media, allowed_type)
                            for allowed_type in allowed_media_types
                        ) and not isinstance(media, Action):
                            aprint(
                                f"Warning: No converter found for non-allowed media of type: {type(media)}"
                            )

            converted_messages.append(new_message)

        # If conversion happened, call convert recursively
        if recursive and conversion_happened:
            return self.convert(converted_messages, allowed_media_types, recursive=True)

        return converted_messages
