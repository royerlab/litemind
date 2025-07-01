import copy
import fnmatch
import os
from string import Formatter
from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Sequence, Set, Type, Union

from arbol import aprint
from pydantic import BaseModel

from litemind.agent.messages.actions.tool_call import ToolCall
from litemind.agent.messages.actions.tool_use import ToolUse
from litemind.agent.messages.message_block import MessageBlock
from litemind.media.media_base import MediaBase
from litemind.media.types.media_action import Action
from litemind.media.types.media_audio import Audio
from litemind.media.types.media_code import Code
from litemind.media.types.media_document import Document
from litemind.media.types.media_file import File
from litemind.media.types.media_image import Image
from litemind.media.types.media_json import Json
from litemind.media.types.media_object import Object
from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text
from litemind.media.types.media_video import Video
from litemind.utils.extract_archive import extract_archive
from litemind.utils.file_types.file_extensions import OTHER_BINARY_EXTS
from litemind.utils.file_types.file_types import (
    has_extension,
    is_archive_file,
    is_audio_file,
    is_document_file,
    is_executable_file,
    is_image_file,
    is_prog_code_file,
    is_script_file,
    is_text_file,
    is_video_file,
    is_web_file,
)
from litemind.utils.folder_description import file_info_header, generate_tree_structure
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path
from litemind.utils.uri_utils import is_uri


class Message:
    def __init__(
        self,
        text: Optional[str] = None,
        json: Optional[str] = None,
        obj: Optional[BaseModel] = None,
        image: Optional[str] = None,
        audio: Optional[str] = None,
        video: Optional[str] = None,
        document: Optional[str] = None,
        table: Optional[str] = None,
        role: str = "user",
    ):
        """
        Create a new message.

        Parameters
        ----------
        role: str
            The role of the message (e.g., 'system', 'user', 'assistant').
        """

        # Check that the role is a string that is either: 'system', 'user', or 'assistant':
        if not isinstance(role, str):
            raise ValueError(f"Role must be a string, not {type(role)}")
        if role not in ["system", "user", "assistant", "tool"]:
            raise ValueError(
                f"Role must be 'system', 'user', or 'assistant', not {role}"
            )

        # Set the role of the message:
        self.role = role
        self.blocks: List[MessageBlock] = []

        # Append the text block if text is provided
        if text:
            self.append_text(text)
        if json:
            self.append_json(json)
        if obj:
            self.append_object(obj)
        if image:
            self.append_image(image)
        if audio:
            self.append_audio(audio)
        if video:
            self.append_video(video)
        if document:
            self.append_document(document)
        if table:
            self.append_table(table)

    def copy(self):
        """
        Create a copy of the message.

        Returns
        -------
        Message
            A copy of the message.
        """
        new_message = Message(role=self.role)
        new_message.blocks = self.blocks.copy()
        return new_message

    def __deepcopy__(self, memo):
        """
        Create a deep copy of the message. The blocks are also deep copied.

        Returns
        -------
        Message
            A deep copy of the message.
        """

        new_message = Message(role=self.role)
        new_message.blocks = [copy.deepcopy(block, memo) for block in self.blocks]
        return new_message

    def append_block(self, block: MessageBlock) -> MessageBlock:
        """
        Append a message block to the message.

        Parameters
        ----------
        block : MessageBlock
            The message block to append.
        """
        self.blocks.append(block)
        return block

    def append_blocks(self, blocks: Union["Message", Sequence[MessageBlock]]):
        """
        Append a list of message blocks to the message.

        Parameters
        ----------
        blocks : Sequence[MessageBlock]
            The message blocks to append.
        """

        # If the blocks are from another message, extract the blocks:
        if isinstance(blocks, Message):
            blocks = blocks.blocks

        # Append the blocks to the message:
        self.blocks.extend(blocks)

    def insert_block(
        self, block: MessageBlock, block_before: MessageBlock
    ) -> MessageBlock:
        """
        Append a message block to the message.

        Parameters
        ----------
        block : MessageBlock
            The message block to append.
        block_before : MessageBlock
            The message block to insert after.
        """

        # find the index of the block_before:
        index = self.blocks.index(block_before)

        # insert the block after the block_before:
        self.blocks.insert(index + 1, block)

        return block

    def insert_blocks(
        self, blocks: List[MessageBlock], block_before: MessageBlock
    ) -> List[MessageBlock]:
        """
        Insert a list of message blocks after a block in the current message.

        Parameters
        ----------
        blocks : List[MessageBlock]
            The message blocks to insert.
        block_before : MessageBlock
            The block to insert the message after.
        """

        # insert the blocks after the block_before:
        for block in blocks:
            self.insert_block(block, block_before)
            block_before = block

        return blocks

    def insert_message(
        self, message: "Message", block_before: MessageBlock
    ) -> "Message":
        """
        Insert a message after a block in the current message.

        Parameters
        ----------
        message : Message
            The message to insert.
        block_before : MessageBlock
            The block to insert the message after.
        """

        self.insert_blocks(message.blocks, block_before)

        return message

    def append_media(self, media: MediaBase, **kwargs) -> MessageBlock:
        """
        Append a media block to the message.

        Parameters
        ----------
        media : MediaBase
            The media block to append.
        kwargs : dict
            Additional attributes for the media block.
        """

        # Append the media block:
        return self.append_block(MessageBlock(media=media, **kwargs))

    def append_text(self, text: str, **kwargs) -> MessageBlock:
        """
        Append text to the message.

        Parameters
        ----------
        text : str
            The text to append to the message.
        kwargs: dict
            Additional attributes for the text block.

        """

        # Append the text block:
        return self.append_block(MessageBlock(media=Text(text=text), **kwargs))

    def append_templated_text(self, template: str, **replacements) -> MessageBlock:
        """
        Replace all `{placeholders}` in *template* with values supplied
        via **replacements and adds resulting text as a new text block.

        Parameters
        ----------
        template:
            The template string containing `{name}` placeholders.

        **replacements:
            Keyword arguments whose names match the placeholders.

        Returns
        -------
        str
            The fully formatted string.

        Raises
        ------
        KeyError
            If one or more placeholders are missing a replacement.
        """
        # Discover every placeholder that appears in *template*
        required: set[str] = {
            field_name
            for _, field_name, _, _ in Formatter().parse(template)
            if field_name  # skip literal braces
        }

        missing = required.difference(replacements)
        if missing:
            raise KeyError(f"Missing replacement(s) for: {', '.join(sorted(missing))}")

        # MappingProxyType gives us an *immutable* view, guarding against
        # accidental mutation inside `format_map`.
        safe_mapping: Mapping[str, Any] = MappingProxyType(replacements)

        # Use `format_map` so that **replacements is not re-copied.
        filled_template = template.format_map(safe_mapping)

        # Append the filled template as a text block:
        return self.append_text(filled_template)

    def append_thinking(self, thinking_text: str, **kwargs) -> MessageBlock:
        """
        Append a thinking block to the message.

        Parameters
        ----------
        thinking_text: str
            The text to append to the message.
        kwargs: dict
            Additional attributes for the thinking block.

        Returns
        -------
        MessageBlock
            The thinking block appended to the message.

        """

        # Check that it is really a string:
        if not isinstance(thinking_text, str):
            raise ValueError(f"Text must be a string, not {type(thinking_text)}")

        # Add 'thinking' to the attributes:
        kwargs["thinking"] = True

        # Append the thinking block:
        return self.append_block(MessageBlock(media=Text(text=thinking_text), **kwargs))

    def append_json(
        self, json_str: str, source: Optional[str] = None, **kwargs
    ) -> MessageBlock:
        """
        Append json to the message.

        Parameters
        ----------
        json_str : str
            The structured text to append to the message.
        source : Optional[str]
            The source of the json (e.g., file path or url).
        kwargs: dict
            Additional attributes for the json block.

        """

        # Append the json block:
        return self.append_block(
            MessageBlock(media=Json(json_str), source=source, **kwargs)
        )

    def append_code(
        self, code: str, lang: str = "python", source: Optional[str] = None, **kwargs
    ) -> MessageBlock:
        """
        Append code to the message.

        Parameters
        ----------
        code : str
            The code to append to the message.
        lang : str
            The language of the code (default: 'python').
        source : Optional[str]
            The source of the code (e.g., file path or url).
        kwargs: dict
            Additional attributes for the json block.
        """

        # Append the code block:
        return self.append_block(
            MessageBlock(media=Code(code=code, lang=lang), source=source, **kwargs)
        )

    def append_object(
        self, obj: BaseModel, source: Optional[str] = None, **kwargs
    ) -> MessageBlock:
        """
        Append a Pydantic object to the message.

        Parameters
        ----------
        obj : BaseModel
            The object to append to the message.
        source : Optional[str]
            The source of the object (e.g., file path or url).
        kwargs: dict
            Additional attributes for the json block.
        """

        # Append the object block:
        return self.append_block(MessageBlock(media=Object(obj), **kwargs))

    def append_image(
        self, image_uri: str, source: Optional[str] = None, **kwargs
    ) -> MessageBlock:
        """
        Append an image to the message.

        Parameters
        ----------
        image_uri : str
            The image to append.
        source : Optional[str]
            The source of the image (e.g., file path or url).
        kwargs: dict
            Additional attributes for the image block.
        """

        # Append the image block:
        return self.append_block(
            MessageBlock(media=Image(uri=image_uri), source=source, **kwargs)
        )

    def append_audio(
        self, audio_uri: str, source: Optional[str] = None, **kwargs
    ) -> MessageBlock:
        """
        Append an audio to the message.

        Parameters
        ----------
        audio_uri : str
            The audio to append.
        source : Optional[str]
            The source of the audio (e.g., file path or url).
        kwargs : dict
            Additional attributes for the audio block.

        """

        # Append the audio block:
        return self.append_block(
            MessageBlock(media=Audio(uri=audio_uri), source=source, **kwargs)
        )

    def append_video(
        self, video_uri: str, source: Optional[str] = None, **kwargs
    ) -> MessageBlock:
        """
        Append a video to the message.

        Parameters
        ----------
        video_uri : str
            The video to append.
        source : Optional[str]
            The source of the video (e.g., file path or url).
        kwargs : dict
            Additional attributes for the video block.
        """

        # Append the video block:
        return self.append_block(
            MessageBlock(Video(uri=video_uri), source=source, **kwargs)
        )

    def append_document(
        self, document_uri: str, source: Optional[str] = None
    ) -> MessageBlock:
        """
        Append a document to the message.

        Parameters
        ----------
        document_uri : str
            The document to append.
        source : Optional[str]
            The source of the document (e.g., file path or url).
        """

        # Append the document block:
        return self.append_block(
            MessageBlock(Document(uri=document_uri), source=source)
        )

    def append_table(
        self, table: Union[str, "ndarray", "DataFrame"], source: Optional[str] = None
    ) -> MessageBlock:
        """
        Append a table to the message.

        Parameters
        ----------
        table : Union[str, ndarray, DataFrame]
            The table to append. Can be a URI of a local or remote file, a pandas DataFrame, or a numpy array.
        source : Optional[str]
            The source of the table (e.g., file path or url).
        """

        # Check that it is a numpy array or a pandas DataFrame:
        from numpy import ndarray
        from pandas import DataFrame

        # If table is an URI of table file (csv, tsv, xls, xlsx, etc.) download file and load it as a pandas DataFrame:
        if isinstance(table, str) and is_uri(table):

            # source is URI that the table comes from:
            source = table

            # Create a table block from the URI:
            block = MessageBlock(Table(table), source=source)

        elif isinstance(table, ndarray):

            # Convert numpy array to DataFrame:
            table = DataFrame(table)

            # Create a table block from the DataFrame:
            block = MessageBlock(Table.from_dataframe(table), source=source)

        elif isinstance(table, DataFrame):

            # Create a table block from the DataFrame:
            block = MessageBlock(Table.from_dataframe(table), source=source)

        else:
            raise ValueError(
                f"Table must be a numpy array or a pandas DataFrame, not {type(table)}"
            )

        # Append the table block:
        return self.append_block(block)

    def append_file(self, file_uri: str, source: Optional[str] = None) -> MessageBlock:
        """
        Append a file to the message.

        Parameters
        ----------
        file_uri : str
            The file to append.
        source : Optional[str]
            The source of the file (e.g., file path or url).
        """

        # Check that the file is a local file:
        if not file_uri.startswith("file://"):
            # Raise an exception if the file URI is not a local file:
            raise ValueError(
                f"File URI must be a local file URI that starts with 'file://', not '{file_uri}'"
            )

        # Remove the 'file://' prefix to get the local file path:
        file_uri_no_prefix = file_uri[7:]  # Remove 'file://' prefix

        # Check that the file URI is valid:
        if not os.path.exists(file_uri_no_prefix):
            raise ValueError(f"File '{file_uri_no_prefix}' does not exist.")

        # Append the file block:
        return self.append_block(MessageBlock(media=File(uri=file_uri), source=source))

    def append_folder(
        self,
        folder: str,
        depth: Optional[int] = None,
        allowed_extensions: List[str] = None,
        excluded_files: List[str] = None,
        all_archive_files: bool = False,
        include_hidden_files: bool = False,
        append_tree_structure: bool = True,
        date_and_times: bool = False,
        file_sizes: bool = True,
    ):
        """
        Append a folder to the message.

        Parameters
        ----------
        folder : str
            The folder to append.
        depth : Optional[int]
            The depth to traverse the folder.
            If None then there is no depth limits (default: None).
        allowed_extensions : List[str]
            The list of allowed file extensions (default: None).
        excluded_files : List[str]
            The list of excluded files (default: None).
        all_archive_files: bool
            Whether to include all files in archives,
            disregarding the depth of files in the archives (default: False).
        include_hidden_files: bool
            Whether to include hidden files (starting with '.') (default: False).
        append_tree_structure: bool
            Whether to append the tree structure of the folder to the message (default: True).
        date_and_times: bool
            Whether to include date and time information in the file info header (default: False).
        file_sizes: bool
            Whether to include file sizes in the file info header (default: True).
        """

        # if depth is zero, then we do not traverse the folder:
        if depth == 0:
            self.append_text(
                f"Folder '{folder}' is not traversed because depth is set to 0."
            )
            return

        # Expand folder string into an absolute path:
        folder = os.path.abspath(folder)

        # If folder does not exist, then append a message that explains that:
        if not os.path.exists(folder):
            self.append_text(f"Folder '{folder}' does not exist.")
            return

        # if folder is empty (does not contain any files or folders) then append a message that explains that:
        if os.path.isdir(folder) and not os.listdir(folder):
            self.append_text(f"Folder '{folder}' is empty.")
            return

        # If folder is a file, then append the file:
        if os.path.isfile(folder):
            # Append the file to the message:
            self.append_text(f"File '{folder}' is not a folder.")
            return

        if append_tree_structure:
            # 1) Generate and append the directory tree (with sizes, no timestamps).
            tree_structure = generate_tree_structure(
                folder,
                allowed_extensions=allowed_extensions,
                excluded_files=excluded_files,
                include_hidden_files=include_hidden_files,
                depth=depth,
            )

            # Append the tree structure to the message:
            self.append_text(f"Directory structure:\n{tree_structure}")

        # 2) Get immediate contents of the folder
        try:
            folder_contents = os.listdir(folder)
            files = []
            dirs = []

            # Separate files and directories
            for item in folder_contents:
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    files.append(item)
                elif os.path.isdir(item_path):
                    dirs.append(item)

            # Process files
            for file_name in sorted(files):
                # Skip hidden files if include_hidden_files is False
                if not include_hidden_files and (
                    file_name.startswith(".") or file_name.startswith("__")
                ):
                    continue

                # Skip files that are excluded:
                if excluded_files and any(
                    fnmatch.fnmatch(file_name, p) for p in excluded_files
                ):
                    aprint(f"File '{file_name}' is excluded.")
                    continue

                # Only keep files with allowed extensions:
                if allowed_extensions and not any(
                    file_name.endswith(ext) for ext in allowed_extensions
                ):
                    aprint(
                        f"File '{file_name}' is not allowed by the extensions filter."
                    )
                    continue

                # Get the file path and URI
                file_path = os.path.join(folder, file_name)
                file_uri = "file://" + file_path

                # if file is empty then just append 'Empty file' message:
                if os.stat(file_path).st_size == 0:
                    header = file_info_header(
                        file_path,
                        "Empty",
                        date_and_times=date_and_times,
                        file_sizes=file_sizes,
                    )
                    self.append_text(header + "\n")
                elif is_prog_code_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Prog. Lang. Code",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_document(file_uri, source=file_path)
                elif is_script_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Script",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_document(file_uri, source=file_path)
                elif is_web_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Web",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_document(file_uri, source=file_path)
                elif is_text_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Text",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_document(file_uri, source=file_path)
                elif is_image_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Image",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_image(file_uri, source=file_path)
                elif is_audio_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Audio",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_audio(file_uri, source=file_path)
                elif is_video_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Video",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_video(file_uri, source=file_path)
                elif is_document_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Document",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_document(file_uri, source=file_path)
                elif is_archive_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Archive (depth={depth})",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    next_depth = None if all_archive_files or not depth else depth - 1
                    self.append_archive(file_uri, next_depth)
                elif is_executable_file(file_path):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Executable",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_file(file_uri, source=file_path)
                elif has_extension(file_path, OTHER_BINARY_EXTS):
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Binary",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_file(file_path, source=file_path)
                else:
                    self.append_text(
                        file_info_header(
                            file_path,
                            "Other",
                            date_and_times=date_and_times,
                            file_sizes=file_sizes,
                        )
                    )
                    self.append_file(file_path, source=file_path)

            # Process subdirectories if depth allows
            if depth is None or depth > 0:
                next_depth = None if depth is None else depth - 1
                for folder_name in sorted(dirs):
                    # Skip hidden directories if include_hidden_files is False
                    if not include_hidden_files and (
                        folder_name.startswith(".") or folder_name.startswith("__")
                    ):
                        continue

                    # Skip directories that are excluded
                    if excluded_files and any(
                        fnmatch.fnmatch(folder_name, p) for p in excluded_files
                    ):
                        aprint(f"Folder '{folder_name}' is excluded.")
                        continue

                    self.append_text(f"\n###### Sub-folder: {folder_name}\n")
                    self.append_folder(
                        folder=os.path.join(folder, folder_name),
                        depth=next_depth,
                        allowed_extensions=allowed_extensions,
                        excluded_files=excluded_files,
                        all_archive_files=all_archive_files,
                        include_hidden_files=include_hidden_files,
                        append_tree_structure=False,
                    )
        except Exception as e:
            # print stack trace for debugging:
            import traceback

            traceback.print_exc()
            # If there is an error accessing the folder, append an error message:
            self.append_text(f"Error accessing folder '{folder}': {str(e)}")

    def append_archive(self, archive_uri: str, depth: int = -1):
        """
        Append an archive to the message.
        Parameters
        ----------
        archive_uri: str
            The archive's uri to append.
        depth: int
            The depth to traverse the archive (default: -1).


        """

        try:
            # Normalize the archive URI:
            archive_local_path = uri_to_local_file_path(archive_uri)

            # extract archive to temporary folder:
            temp_folder = extract_archive(archive_local_path)

            # append the extracted folder to the message:
            self.append_text(
                f"\n##### Contents of archive: {archive_uri} decompressed into folder: "
                + temp_folder
                + "\n"
            )

            # append the extracted folder to the message:
            self.append_folder(temp_folder, depth)
        except Exception as e:
            # If there is an error extracting the archive, append an error message:
            self.append_text(
                f"Error while attempting to extract files from archive '{archive_uri}': {str(e)}"
            )
            self.append_file(archive_uri, source=archive_uri)
            return

    def append_tool_call(
        self, tool_name: str, arguments: dict, id: str
    ) -> MessageBlock:
        """
        Add a tool call to the message. This is typically part of an assistant's response.

        Parameters
        ----------
        tool_name: str
            The name of the tool.
        arguments: dict
            The arguments used with the tool.
        id: str
            The id of the tool use message.
        """

        # Create a new tool use message block:
        block = MessageBlock(media=Action(ToolCall(tool_name, arguments, id)))

        # Append the tool use block to the message:
        return self.append_block(block)

    def append_tool_use(
        self, tool_name: str, arguments: dict, result: Any, id: str
    ) -> MessageBlock:
        """
        Add a tool use to the message. This is typically part of a user's input in reply of an assistant's tool call.

        Parameters
        ----------
        tool_name: str
            The name of the tool.
        arguments: dict
            The arguments used with the tool.
        result: Any
            The result of the tool use.
        id: str
            The id of the tool use message.
        """

        # Create a new tool use message block:
        block = MessageBlock(media=Action(ToolUse(tool_name, arguments, result, id)))

        # Append the tool use block to the message:
        return self.append_block(block)

    def extract_markdown_block(
        self, filters: Union[str, List[str]], remove_quotes=True
    ) -> List[MessageBlock]:
        """
        Extract markdown blocks from the message that contain the given filter string.
        Parameters
        ----------
        filters: Union[str, List[str]]
            The filter string to search for in the markdown blocks. Can be a singleton string for convenience.
        remove_quotes: bool
            Whether to remove the quotes from the markdown block content (default: True).

        Returns
        -------
        List[MessageBlock]
            The markdown blocks that contain the filter string.

        """
        # Extract markdown blocks from the message that contain the given filter string.

        # If the filters parameter is a string, convert it to a list:
        if isinstance(filters, str):
            filters = [filters]

        # Initialize the list of markdown blocks:
        markdown_blocks: List[MessageBlock] = []

        # Iterate over the blocks in the message:
        for block in self.blocks:

            # Check if the block is a text block and contains the filter string:
            if isinstance(block.media, Text) and any(
                f in block.get_content() for f in filters
            ):
                # Parse the string to extract text strings in the form: "```... ```"
                # and create a new markdown block for each one.
                text = "" + block.get_content()

                # Extract markdown blocks from the text:
                while "```" in text:
                    start = text.find("```")
                    end = text.find("```", start + 3)
                    if end == -1:
                        break
                    if remove_quotes:
                        # find the first '\n' after the first ``` and set start after '\n':
                        start = text.find("\n", start + 3) + 1
                        # add the markdown block to the list:
                        markdown_blocks.append(
                            MessageBlock(media=Text(text[start:end]))
                        )
                    else:
                        # add the markdown block to the list:
                        markdown_blocks.append(
                            MessageBlock(media=Text(text[start : end + 3]))
                        )

                    # remove the markdown block from the text:
                    text = text[end + 3 :]

        return markdown_blocks

    def list_media_types(self) -> Sequence[Type[MediaBase]]:
        """
        List the media classes present in the message blocks.

        Returns
        -------
        Sequence[Type[MediaBase]]
            A list of media classes present in the message blocks.
        """
        media_classes: Set[Type[MediaBase]] = set()
        for block in self.blocks:
            if isinstance(block.media, MediaBase):
                media_classes.add(type(block.media))
        return list(media_classes)

    @staticmethod
    def list_media_types_in(messages: Sequence["Message"]) -> Sequence[Type[MediaBase]]:
        """
        List the media classes present in the given messages.

        Parameters
        ----------
        messages : Sequence[Message]
            The messages to analyze.

        Returns
        -------
        Sequence[Type[MediaBase]]
            A list of media classes present in the messages.
        """
        media_classes: Set[Type[MediaBase]] = set()
        message: Message
        for message in messages:
            media_type: Type[MediaBase]
            for media_type in message.list_media_types():
                media_classes.add(media_type)
        return list(media_classes)

    def convert_media(
        self,
        allowed_media_types: Optional[List[Type[MediaBase]]] = None,
        media_converter: Optional["MediaConverter"] = None,
    ) -> "Message":
        """
        Convert the media in the message blocks to the allowed media types.

        Parameters
        ----------
        allowed_media_types : Optional[List[Type[MediaBase]]]
            The list of allowed media types to convert to. If None, only text media types are allowed.
        media_converter: Optional[MediaConverter]
            The media converter to use for conversion. If None, a default media converter is used.

        Returns
        -------
        Message
            A new message with the converted media blocks.
        """

        if allowed_media_types is None:
            # If no allowed media types are provided, default to text media type:
            allowed_media_types = [Text]

        # If no media converter is provided, instantiate a default one:
        if media_converter is None:
            from litemind.media.conversion.media_converter import MediaConverter

            # Instantiate the media converter:
            media_converter = MediaConverter()

            # Add the media converter supporting APIs. This converter can use any model from the API that supports the required features.
            media_converter.add_default_converters()

        # Convert the message using the media converter:
        converted_messages = media_converter.convert(
            [self], allowed_media_types=allowed_media_types
        )

        # Get the converted message:
        converted_message = converted_messages[0]

        return converted_message

    def compress_text(
        self, text_compressor: Optional["TextCompressor"] = None
    ) -> "Message":
        """
        Compress the text in the message blocks by using TextCompressor
        to reduce the size of the text blocks.

        Parameters
        ----------
        text_compressor: Optional[TextCompressor]
            The text compressor to use for compression. If None, a default text compressor is used.

        Returns
        -------
        Message
            A new message with the compressed text blocks.
        """

        # Create a new message to hold the compressed text blocks:
        compressed_message = Message(role=self.role)

        # Create a text compressor:
        if text_compressor is None:
            # If no text compressor is provided, instantiate a default one:
            from litemind.utils.text_compressor import TextCompressor

            text_compressor = TextCompressor()

        # Iterate over the blocks in the message:
        for block in self.blocks:
            if isinstance(block.media, Text):
                # Compress the text block:
                compressed_text = text_compressor.compress(block.media.text)

                if len(compressed_text) > 0:
                    # Append the compressed text block to the compressed message:
                    compressed_message.append_text(compressed_text, **block.attributes)
            else:
                # Append the block as is if it is not a text block:
                compressed_message.append_block(block)

        return compressed_message

    def __getitem__(self, index: int) -> Union[MessageBlock, None]:
        """
        Get the message block at the given index.

        Parameters
        ----------
        index : int
            The index of the message block to get.

        Returns
        -------
        MessageBlock or None
            The message block at the given index, or None if the index is out of range.
        """
        return self.blocks[index]

    def __len__(self) -> int:
        """
        Return the number of blocks in the message.

        Returns
        -------
        int
            The number of blocks in the message.
        """
        return len(self.blocks)

    def __contains__(self, item: str) -> bool:
        """
        Check if the given string is in the message blocks.

        Parameters
        ----------
        item : str
            The string to check for.

        Returns
        -------
        bool
            True if the string is found, False otherwise.
        """
        for block in self.blocks:
            if item in str(block):
                return True
        return False

    def has(self, block_type: Type[MediaBase]):
        """
        Check if the given block type is in the message blocks.

        Parameters
        ----------
        block_type : Type[MediaBase]
            The block type to check for.

        Returns
        -------
        bool
            True if the block type is found, False otherwise.
        """
        for block in self.blocks:
            if isinstance(block.media, block_type):
                return True
        return False

    def __str__(self) -> str:
        """
        Return the message as a string.
        Returns
        -------
        str
            The message as a string.
        """
        message_string = f"*{self.role}*:\n"
        for block in self.blocks:
            message_string += str(block) + "\n"
        return message_string

    def lower(self) -> str:
        """
        Return the message as a string in lowercase.
        Returns
        -------
        str
            The message as a string in lowercase.
        """
        return self.__str__().lower()

    def to_plain_text(self) -> str:
        """
        Return the message as plain text.

        Returns
        -------
        str
            The message as plain text.

        """
        plain_text = ""
        for block in self.blocks:
            plain_text += str(block.media) + "\n"
        return plain_text

    def to_markdown(self, media_converter: Optional["MediaConverter"] = None):

        if media_converter is None:
            # If no media converter is provided, instantiate a default one:
            from litemind.media.conversion.media_converter import MediaConverter

            # Instantiate the media converter:
            media_converter = MediaConverter()

            # Add the media converter supporting APIs. This converter can use any model from the API that supports the required features.
            media_converter.add_default_converters()

        # Convert the document to markdown:
        from litemind.media.types.media_text import Text

        markdown_text = media_converter.convert(
            messages=[self], allowed_media_types=[Text]
        )

        # Get the converted message:
        converted_message = markdown_text[0]

        # Get the markdown string from the converted message blocks:
        markdown_string = converted_message.to_plain_text()

        return markdown_string

    def report(self, as_string=True) -> Union[dict, str]:
        """
        Generate a report with statistics about the message blocks.

        Parameters
        ----------
        as_string : bool
            Whether to return the report as a formatted string (True) or a dictionary (False).

        Returns
        -------
        Union[dict, str]
            A report containing statistics about the message blocks.
        """
        # Initialize report data
        report_data = {
            "total_blocks": len(self.blocks),
            "blocks_by_type": {},
            "longest_blocks": {},
            "average_block_size": 0,
            "average_by_type": {},
            "total_characters": 0,
            "role": self.role,
        }

        # Collect data for each block
        block_lengths = []
        lengths_by_type = {}

        for block in self.blocks:
            # Get media type name
            media_type = type(block.media).__name__

            # Count blocks by type
            if media_type not in report_data["blocks_by_type"]:
                report_data["blocks_by_type"][media_type] = 0
                lengths_by_type[media_type] = []

            report_data["blocks_by_type"][media_type] += 1

            # Calculate block length (character count)
            block_str = str(block)
            block_length = len(block_str)
            block_lengths.append(block_length)
            lengths_by_type[media_type].append(block_length)

            report_data["total_characters"] += block_length

            # Check if this is the longest block of its type
            if (
                media_type not in report_data["longest_blocks"]
                or block_length > report_data["longest_blocks"][media_type]["length"]
            ):
                # Preview the content of the block:
                preview_block_str = (
                    block_str[:100] if len(block_str) > 100 else block_str
                )

                # Replace newlines with emoji characters that look line new lines for better preview:
                preview_block_str = preview_block_str.replace("\n", "↩️")

                report_data["longest_blocks"][media_type] = {
                    "length": block_length,
                    "content_preview": preview_block_str + "...",
                }

        # Calculate averages
        if report_data["total_blocks"] > 0:
            report_data["average_block_size"] = (
                report_data["total_characters"] / report_data["total_blocks"]
            )

        # Calculate average by type
        for media_type, lengths in lengths_by_type.items():
            if lengths:
                report_data["average_by_type"][media_type] = sum(lengths) / len(lengths)

        # Add additional statistics
        if block_lengths:
            report_data["min_block_size"] = min(block_lengths)
            report_data["max_block_size"] = max(block_lengths)
            report_data["median_block_size"] = sorted(block_lengths)[
                len(block_lengths) // 2
            ]

        if not as_string:
            return report_data

        # Format report as string
        report_str = f"Message Report (Role: {self.role})\n"
        report_str += f"Total blocks: {report_data['total_blocks']}\n"
        report_str += f"Total characters: {report_data['total_characters']}\n\n"

        report_str += "Blocks by type:\n"
        for media_type, count in sorted(report_data["blocks_by_type"].items()):
            report_str += f"  {media_type}: {count} blocks\n"

        report_str += f"\nAverage block size: {report_data['average_block_size']:.2f} characters\n"

        report_str += "\nAverage size by type:\n"
        for media_type, avg in sorted(report_data["average_by_type"].items()):
            report_str += f"  {media_type}: {avg:.2f} characters\n"

        report_str += "\nLongest blocks by type:\n"
        for media_type, info in sorted(report_data["longest_blocks"].items()):
            report_str += f"  {media_type}: {info['length']} characters\n"
            report_str += f"    Preview: {info['content_preview']}\n"

        if "min_block_size" in report_data:
            report_str += (
                f"\nMin block size: {report_data['min_block_size']} characters\n"
            )
            report_str += (
                f"Max block size: {report_data['max_block_size']} characters\n"
            )
            report_str += (
                f"Median block size: {report_data['median_block_size']} characters\n"
            )

        return report_str

    def __repr__(self) -> str:
        """
        Return the message as a string.
        Returns
        -------
        str
            The message as a string.
        """
        return str(self)

    def __iter__(self):
        """
        Iterate over the message blocks.
        """
        return iter(self.blocks)

    def __hash__(self):
        """
        Return the hash of the message.
        Returns
        -------
        int
            The hash of the message.
        """
        return hash((self.role, tuple(self.blocks)))

    def __eq__(self, other: "Message") -> bool:
        """
        Compare two messages for equality.
        Returns
        -------
        bool
            True if the messages are equal, False otherwise.
        """

        # Check if is a Message:
        if not isinstance(other, Message):
            raise TypeError(f"Cannot compare {type(other)} with {type(self)}")

        # Check if the roles are the same:
        if self.role != other.role:
            return False

        # Check that there is the same number of blocks:
        if len(self.blocks) != len(other.blocks):
            return False

        # Check that the blocks are the same:
        for block1, block2 in zip(self.blocks, other.blocks):
            if block1 != block2:
                return False

        # If all checks pass, the messages are equal:
        return True

    def __ne__(self, other: "Message") -> bool:
        """
        Compare two messages for inequality.
        Returns
        -------
        bool
            True if the messages are not equal, False otherwise.
        """
        return not self == other

    def __add__(self, other: Union["Message", MessageBlock]) -> "Message":
        """
        Concatenate two messages.

        Parameters
        ----------
        other : Union[Message, MessageBlock]
            The message or message block to concatenate.

        Returns
        -------
        Message
            The concatenated message.
        """

        # Create a new message as copy of the current message:
        new_message = self.copy()

        if isinstance(other, Message):
            # Concatenate the blocks of the other message to the new message:
            new_message.blocks += other.blocks
        elif isinstance(other, MessageBlock):
            # Append the message block to the new message:
            new_message.blocks.append(other)
        else:
            # Raise an error if the other object is not a Message or MessageBlock:
            raise ValueError(
                f"Cannot concatenate message with {type(other)}. Only Message or MessageBlock are allowed."
            )

        # Return the new message:
        return new_message

    def __iadd__(self, other: Union["Message", MessageBlock]) -> "Message":
        """
        Concatenate another message to the current message.

        Parameters
        ----------
        other : Union[Message, MessageBlock]
            The message or message block to concatenate.

        Returns
        -------
        Message
            The concatenated message.
        """
        if isinstance(other, Message):
            # Concatenate the blocks of the other message to the current message:
            self.blocks += other.blocks
        elif isinstance(other, MessageBlock):
            # Append the message block to the current message:
            self.blocks.append(other)
        else:
            # Raise an error if the other object is not a Message or MessageBlock:
            raise ValueError(
                f"Cannot concatenate message with {type(other)}. Only Message or MessageBlock are allowed."
            )

        # Return the current message:
        return self

    def __radd__(self, other: Union["Message", MessageBlock]) -> "Message":
        """
        Concatenate the current message to another message.

        Parameters
        ----------
        other : Union[Message, MessageBlock]
            The message or message block to concatenate.

        Returns
        -------
        Message
            The concatenated message.
        """

        # Create a new message as copy of the current message:
        new_message = other.copy()

        if isinstance(other, Message):
            # Concatenate the blocks of the current message to the new message:
            new_message.blocks += self.blocks
        elif isinstance(other, MessageBlock):
            # Append the message block to the new message:
            new_message.blocks.append(self)
        else:
            # Raise an error if the other object is not a Message or MessageBlock:
            raise ValueError(
                f"Cannot concatenate message with {type(other)}. Only Message or MessageBlock are allowed."
            )
        return new_message
