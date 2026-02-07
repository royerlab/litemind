from typing import List, Optional, Sequence, Set, Type

from arbol import aprint

from litemind.agent.messages.message import Message
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.conversion.converters.base_converter import BaseConverter
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_types import all_media_types


class MediaConverter:
    """Pipeline for converting messages with arbitrary media to a restricted set of allowed types.

    Maintains an ordered list of ``BaseConverter`` instances. When
    converting, the first converter that can handle a given media takes
    precedence. Conversion is applied recursively by default so that
    intermediate types (e.g. Table -> Text) are fully resolved.
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
        """Register the standard set of converters.

        Adds converters for NdImage, Table, Object, Code, Json, and
        optionally Video (requires ffmpeg), Document (requires pymupdf
        and/or docling), and Audio (requires local Whisper). Converters
        are only added when their dependencies are available.

        Parameters
        ----------
        convert_audio : bool, optional
            Include the Whisper-based audio converter. Default is True.
        convert_videos : bool, optional
            Include the ffmpeg-based video converter. Default is True.
        convert_documents : bool, optional
            Include document converters (pymupdf, docling, txt fallback).
            Default is True.

        Returns
        -------
        List[BaseConverter]
            The converters that were actually added.
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
        """Add a converter to the pipeline.

        Parameters
        ----------
        media_converter : BaseConverter
            The converter to add.
        highest_priority : bool, optional
            If True, insert at the front of the list so it takes precedence
            over existing converters. Default is False (append).
        """
        if highest_priority:
            # Add the media converter to the beginning of the list
            self.media_converters.insert(0, media_converter)
        else:
            # Add the media converter to the end of the list
            self.media_converters.append(media_converter)

    def remove_media_converter(self, media_converter: BaseConverter):
        """Remove a converter from the pipeline.

        Parameters
        ----------
        media_converter : BaseConverter
            The converter instance to remove.

        Raises
        ------
        ValueError
            If the converter is not in the pipeline.
        """
        self.media_converters.remove(media_converter)

    def can_convert_within(
        self,
        source_media_type: Type[MediaBase],
        allowed_media_types: Set[Type[MediaBase]],
    ) -> bool:
        """Check whether a source type can be fully converted to allowed types.

        Performs a BFS over the conversion graph to verify that every
        reachable type from *source_media_type* is contained in
        *allowed_media_types*.

        Parameters
        ----------
        source_media_type : Type[MediaBase]
            The media type to start from.
        allowed_media_types : Set[Type[MediaBase]]
            The set of acceptable target types.

        Returns
        -------
        bool
            True if all conversion outputs are within the allowed set.
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
        """Find all media types that can be converted to the allowed set.

        Parameters
        ----------
        allowed_media_types : Set[Type[MediaBase]]
            The set of acceptable target types.

        Returns
        -------
        Set[Type[MediaBase]]
            All media types for which ``can_convert_within`` returns True.
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
        """Convert messages so they contain only allowed media types.

        Iterates over each message block and applies the first matching
        converter. The conversion is **not** performed in place; new
        messages are returned.

        Parameters
        ----------
        messages : List[Message]
            The messages to convert.
        allowed_media_types : Sequence[Type[MediaBase]]
            Media types that should be kept as-is.
        exclude_extensions : Sequence[str], optional
            File extensions to skip during conversion (media with these
            extensions are kept unchanged). Default is None (no exclusions).
        recursive : bool, optional
            If True, reapply conversion until no further changes occur.
            Default is True.

        Returns
        -------
        List[Message]
            New messages containing only allowed (or unconvertible) media.
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
                        for converted_item in converted:
                            new_message.append_block(
                                MessageBlock(
                                    media=converted_item, attributes=attributes
                                )
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
