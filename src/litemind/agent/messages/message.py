import copy
import os
from abc import ABC
from json import loads
from typing import Any, List, Optional, Sequence, Union

from pydantic import BaseModel

from litemind.agent.messages.message_block import MessageBlock
from litemind.agent.messages.message_block_type import BlockType
from litemind.agent.messages.tool_call import ToolCall
from litemind.agent.messages.tool_use import ToolUse
from litemind.agent.utils.folder_description import (
    file_info_header,
    generate_tree_structure,
    is_text_file,
    read_binary_file_info,
    read_file_content,
)
from litemind.utils.extract_archive import extract_archive
from litemind.utils.file_extensions import (
    archive_file_extensions,
    audio_file_extensions,
    document_file_extensions,
    image_file_extensions,
    table_file_extensions,
    video_file_extensions,
)
from litemind.utils.normalise_uri_to_local_file_path import uri_to_local_file_path


class Message(ABC):
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

        # Append the thinking block:
        return self.append_block(
            MessageBlock(block_type="thinking", content=thinking_text, **kwargs)
        )

    def append_text(self, text: str) -> MessageBlock:
        """
        Append text to the message.

        Parameters
        ----------
        text : str
            The text to append to the message.
        """
        # Check that it is really a string:
        if not isinstance(text, str):
            raise ValueError(f"Text must be a string, not {type(text)}")

        # Append the text block:
        return self.append_block(MessageBlock(block_type="text", content=text))

    def append_json(self, json_str: str, source: Optional[str] = None) -> MessageBlock:
        """
        Append json to the message.

        Parameters
        ----------
        json_str : str
            The structured text to append to the message.
        source : Optional[str]
            The source of the json (e.g., file path or url).

        """

        # Check that it is correctly formatted json:
        try:
            # parse json string using json library to check if it correctly formatted:
            parsed_json = loads(json_str)
            assert parsed_json and isinstance(parsed_json, dict)

        except:
            raise ValueError(f"Json must be correctly formatted")

        # Append the json block:
        return self.append_block(
            MessageBlock(block_type="json", content=json_str, source=source)
        )

    def append_code(
        self, code: str, lang: str = "python", source: Optional[str] = None
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
        """

        # Check that it is really a string:
        if not isinstance(code, str):
            raise ValueError(f"Code must be a string, not {type(code)}")

        # Append the code block:
        return self.append_block(
            MessageBlock(block_type="code", content=code, lang=lang, source=source)
        )

    def append_object(self, obj: BaseModel) -> MessageBlock:
        """
        Append a Pydantic object to the message.

        Parameters
        ----------
        obj : BaseModel
            The object to append to the message.
        """

        # Check that it is a Pydantic object:
        if not isinstance(obj, BaseModel):
            raise ValueError(f"Object must be a Pydantic object, not {type(obj)}")

        # Append the object block:
        return self.append_block(MessageBlock(block_type="object", content=obj))

    def append_uri(
        self, uri: str, block_type: str, source: Optional[str] = None
    ) -> MessageBlock:
        """
        Append a URI to the message.

        Parameters
        ----------
        uri : str
            The URI to append.
        block_type : str
            The type of the block (e.g., 'image', 'audio', 'video', 'document')
        source : Optional[str]
            The source of the URI (e.g., file path or url).
        """

        # Check that it is a valid URI:
        if (
            not uri.startswith("http")
            and not uri.startswith("file")
            and not uri.startswith("data")
        ):
            raise ValueError(
                f"Invalid URI: '{uri}' (must start with 'http', 'file', or 'data')"
            )

        # Append the URI block:
        return self.append_block(
            MessageBlock(block_type=block_type, content=uri, source=source)
        )

    def append_image(
        self, image_uri: str, source: Optional[str] = None
    ) -> MessageBlock:
        """
        Append an image to the message.

        Parameters
        ----------
        image_uri : str
            The image to append.
        source : Optional[str]
            The source of the image (e.g., file path or url).
        """

        # Check that it is a valid image URI (including data uri):
        if (
            not image_uri.startswith("http")
            and not image_uri.startswith("file")
            and not image_uri.startswith("data")
        ):
            raise ValueError(
                f"Invalid image URI: '{image_uri}' (must start with 'http', 'file', or 'data')"
            )

        # Append the image block:
        return self.append_uri(image_uri, "image", source=source)

    def append_audio(
        self, audio_uri: str, source: Optional[str] = None
    ) -> MessageBlock:
        """
        Append an audio to the message.

        Parameters
        ----------
        audio_uri : str
            The audio to append.
        source : Optional[str]
            The source of the audio (e.g., file path or url).
        """

        # Check that it is a valid audio URI (including data uri):
        if (
            not audio_uri.startswith("http")
            and not audio_uri.startswith("file")
            and not audio_uri.startswith("data")
        ):
            raise ValueError(
                f"Invalid audio URI: '{audio_uri}' (must start with 'http', 'file', or 'data')"
            )

        # Append the audio block:
        return self.append_uri(audio_uri, "audio", source=source)

    def append_video(
        self, video_uri: str, source: Optional[str] = None
    ) -> MessageBlock:
        """
        Append a video to the message.

        Parameters
        ----------
        video_uri : str
            The video to append.
        source : Optional[str]
            The source of the video (e.g., file path or url).
        """

        # Check that  the file extension is valid:
        if not any(video_uri.endswith(ext) for ext in video_file_extensions):
            raise ValueError(
                f"Invalid video URI: '{video_uri}' (must have a valid video file extension)"
            )

        # Append the video block:
        return self.append_uri(video_uri, "video", source=source)

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

        # Check that the file extension is valid:
        if not any(document_uri.endswith(ext) for ext in document_file_extensions):
            # If it is a remote URL, then it is ok to not see a correct extension!
            if document_uri.startswith("http"):
                pass
            else:
                raise ValueError(
                    f"Invalid document URI: '{document_uri}' (must have a valid document file extension)"
                )

        # Append the document block:
        return self.append_uri(document_uri, "document", source=source)

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

        # If table is an URI of table file (csv, tsv, xls, xlsx, etc.) download file and load it as a pandas DataFrame:
        if isinstance(table, str):

            # source is URI that the table comes from:
            source = table

            # Check that it is a valid table URI:
            if not table.startswith("http") and not table.startswith("file"):
                raise ValueError(
                    f"Invalid table URI: '{table}' (must start with 'http' or 'file')"
                )

            # Check that the file extension is valid:
            if not any(table.endswith(ext) for ext in table_file_extensions):
                raise ValueError(
                    f"Invalid table URI: '{table}' (must have a valid table file extension)"
                )

            # Download the table file to a local temp file using download_table_to_temp_file:
            table = uri_to_local_file_path(table)

            # Detect character encoding:
            import chardet

            with open(table, "rb") as f:
                result = chardet.detect(f.read())

            # Download the table file taking care of the character encoding and errors
            from pandas import read_csv

            table = read_csv(table, encoding=result["encoding"])

        # Check that it is a numpy array or a pandas DataFrame:
        from numpy import ndarray
        from pandas import DataFrame

        if not (isinstance(table, ndarray) or isinstance(table, DataFrame)):
            raise ValueError(
                f"Table must be a numpy array or a pandas DataFrame, not {type(table)}"
            )

        # Append the table block:
        return self.append_block(
            MessageBlock(block_type="table", content=table, source=source)
        )

    def append_folder(
        self,
        folder: str,
        depth: int = None,
        allowed_extensions: List[str] = None,
        excluded_files: List[str] = None,
        all_archive_files: bool = False,
        include_hidden_files: bool = False,
    ) -> MessageBlock:
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
        """

        # Expand folder string into an absolute path:
        folder = os.path.abspath(folder)

        # 1) Generate and append the directory tree (with sizes, no timestamps).
        tree_structure = generate_tree_structure(
            folder,
            allowed_extensions=allowed_extensions,
            excluded_files=excluded_files,
            include_hidden_files=include_hidden_files,
        )

        # Append the tree structure to the message:
        self.append_text(f"Directory structure:\n{tree_structure}")

        # 2) Recursively traverse folder to process each file
        for root, dirs, files in os.walk(folder):

            # Get the folder name without the path:
            folder_name = os.path.basename(root)

            # Skip hidden files if include_hidden_files is False
            if not include_hidden_files and (
                folder_name.startswith(".") or folder_name.startswith("__")
            ):
                continue

            # Skip folders that are excluded:
            if excluded_files and folder_name in excluded_files:
                continue

            for file in files:

                # Skip hidden files if include_hidden_files is False
                if not include_hidden_files and (
                    file.startswith(".") or file.startswith("__")
                ):
                    continue

                # Skip files that are excluded:
                if excluded_files and file in excluded_files:
                    continue

                # Only keep files with allowed extensions:
                if allowed_extensions and not any(
                    file.endswith(ext) for ext in allowed_extensions
                ):
                    continue

                # Get the file path and URI
                file_path = os.path.join(root, file)
                file_uri = "file://" + file_path

                # if file is empty then just append 'Empty file' message:
                if os.stat(file_path).st_size == 0:
                    header = file_info_header(file_path, "Empty")
                    self.append_text(header + "\n")
                elif is_text_file(file_path):
                    header = file_info_header(file_path, "Text")
                    content = read_file_content(file_path)
                    self.append_text(header + content + "\n")
                elif any(file.endswith(ext) for ext in image_file_extensions):
                    header = file_info_header(file_path, "Image")
                    self.append_text(header)
                    self.append_image(file_uri, source=file_path)
                elif any(file.endswith(ext) for ext in audio_file_extensions):
                    header = file_info_header(file_path, "Audio")
                    self.append_text(header)
                    self.append_audio(file_uri, source=file_path)
                elif any(file.endswith(ext) for ext in video_file_extensions):
                    header = file_info_header(file_path, "Video")
                    self.append_text(header)
                    self.append_video(file_uri, source=file_path)
                elif any(file.endswith(ext) for ext in document_file_extensions):
                    header = file_info_header(file_path, "Document")
                    self.append_text(header)
                    self.append_document(file_uri, source=file_path)
                elif any(file.endswith(ext) for ext in archive_file_extensions):
                    header = file_info_header(file_path, "Compressed Archive")
                    self.append_text(header)
                    next_depth = None if all_archive_files or not depth else depth - 1
                    self.append_archive(file_uri, next_depth)
                else:
                    header = file_info_header(file_path, "Binary")
                    size, hex_content = read_binary_file_info(file_path)
                    binary_body = f"First 100 bytes (hex): {hex_content}\n"
                    self.append_text(header + binary_body)

            # If a depth limit is set, only recurse deeper if depth >= 1
            if depth is not None and depth >= 1:
                for d in dirs:
                    self.append_text(f"\n###### Sub-folder: {d}\n")
                    self.append_folder(
                        folder=os.path.join(root, d),
                        depth=depth - 1,
                        allowed_extensions=allowed_extensions,
                        excluded_files=excluded_files,
                        all_archive_files=all_archive_files,
                        include_hidden_files=include_hidden_files,
                    )

    def append_archive(self, archive: str, depth: int = 1):
        """
        Append an archive to the message.
        Parameters
        ----------
        archive: str
            The archive to append.
        depth: int
            The depth to traverse the archive (default: 1).


        """

        # extract archive to temporary folder:
        temp_folder = extract_archive(archive)

        # append the extracted folder to the message:
        self.append_text(
            f"\n##### Contents of archive: {archive} decompressed into folder: "
            + temp_folder
            + "\n"
        )

        # append the extracted folder to the message:
        self.append_folder(temp_folder, depth)

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
        block = MessageBlock(
            block_type=BlockType.Tool, content=ToolCall(tool_name, arguments, id)
        )

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
        block = MessageBlock(
            block_type=BlockType.Tool, content=ToolUse(tool_name, arguments, result, id)
        )

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
            if block.block_type == BlockType.Text and any(
                f in block.content for f in filters
            ):
                # Parse the string to extract text strings in the form: "```... ```"
                # and create a new markdown block for each one.
                text = "" + block.content

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
                            MessageBlock(
                                block_type=BlockType.Text, content=text[start:end]
                            )
                        )
                    else:
                        # add the markdown block to the list:
                        markdown_blocks.append(
                            MessageBlock(
                                block_type=BlockType.Text, content=text[start : end + 3]
                            )
                        )

                    # remove the markdown block from the text:
                    text = text[end + 3 :]

        return markdown_blocks

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
            if item in str(block.content):
                return True
        return False

    def has(self, block_type: BlockType):
        """
        Check if the given block type is in the message blocks.

        Parameters
        ----------
        block_type : BlockType
            The block type to check for.

        Returns
        -------
        bool
            True if the block type is found, False otherwise.
        """
        for block in self.blocks:
            if block.block_type == block_type:
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
            if block.block_type == BlockType.Text:
                plain_text += block.content + "\n"
        return plain_text

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
        if self.role != other.role:
            return False
        if len(self.blocks) != len(other.blocks):
            return False
        for block1, block2 in zip(self.blocks, other.blocks):
            if block1 != block2:
                return False
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

    def __add__(self, other: "Message") -> "Message":
        """
        Concatenate two messages.
        Returns
        -------
        Message
            The concatenated message.
        """
        new_message = self.copy()
        new_message.blocks += other.blocks
        return new_message

    def __iadd__(self, other: "Message") -> "Message":
        """
        Concatenate another message to the current message.
        Returns
        -------
        Message
            The concatenated message.
        """
        self.blocks += other.blocks
        return self

    def __radd__(self, other: "Message") -> "Message":
        """
        Concatenate the current message to another message.
        Returns
        -------
        Message
            The concatenated message.
        """
        new_message = other.copy()
        new_message.blocks += self.blocks
        return new_message
