import os
from abc import ABC
from json import loads
from typing import Optional, List, Union

from pydantic import BaseModel

from litemind.agent.message_block import MessageBlock
from litemind.utils.dowload_table_to_tempfile import download_table_to_temp_file
from litemind.utils.extract_archive import extract_archive
from litemind.utils.file_extensions import image_file_extensions, \
    audio_file_extensions, video_file_extensions, document_file_extensions, \
    archive_file_extensions, table_file_extensions


class Message(ABC):
    def __init__(self,
                 role: str,
                 text: Optional[str] = None,
                 json: Optional[str] = None,
                 obj: Optional[BaseModel] = None,
                 image: Optional[str] = None,
                 audio: Optional[str] = None,
                 video: Optional[str] = None,
                 document: Optional[str] = None,
                 table: Optional[str] = None):
        """
        Create a new message.

        Parameters
        ----------
        role: str
            The role of the message (e.g., 'user' or 'agent').
        """
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

    def append_block(self, block: MessageBlock):
        """
        Append a message block to the message.

        Parameters
        ----------
        block : MessageBlock
            The message block to append.
        """
        self.blocks.append(block)

    def append_text(self, text: str):
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
        self.append_block(MessageBlock(block_type='text', content=text))

    def append_json(self,
                    json_str: str,
                    source: Optional[str] = None):
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
        self.append_block(
            MessageBlock(block_type='json', content=json_str, source=source))

    def append_code(self, code: str,
                    lang: str = 'python',
                    source: Optional[str] = None
                    ):
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
        self.append_block(
            MessageBlock(block_type='code', content=code, lang=lang,
                         source=source))

    def append_object(self, obj: BaseModel):
        """
        Append a Pydantic object to the message.

        Parameters
        ----------
        obj : BaseModel
            The object to append to the message.
        """

        # Check that it is a Pydantic object:
        if not isinstance(obj, BaseModel):
            raise ValueError(
                f"Object must be a Pydantic object, not {type(obj)}")

        # Append the object block:
        self.append_block(MessageBlock(block_type='json', content=obj))

    def append_uri(self,
                   uri: str,
                   block_type: str,
                   source: Optional[str] = None):
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
        if not uri.startswith('http') and not uri.startswith(
                'file') and not uri.startswith('data'):
            raise ValueError(
                f"Invalid URI: '{uri}' (must start with 'http', 'file', or 'data')")

        # Append the URI block:
        self.append_block(
            MessageBlock(block_type=block_type, content=uri, source=source))

    def append_image(self,
                     image_uri: str,
                     source: Optional[str] = None):
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
        if not image_uri.startswith('http') and not image_uri.startswith(
                'file') and not image_uri.startswith('data'):
            raise ValueError(
                f"Invalid image URI: '{image_uri}' (must start with 'http', 'file', or 'data')")

        # Append the image block:
        self.append_uri(image_uri, 'image', source=source)

    def append_audio(self,
                     audio_uri: str,
                     source: Optional[str] = None):
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
        if not audio_uri.startswith('http') and not audio_uri.startswith(
                'file') and not audio_uri.startswith('data'):
            raise ValueError(
                f"Invalid audio URI: '{audio_uri}' (must start with 'http', 'file', or 'data')")

        # Append the audio block:
        self.append_uri(audio_uri, 'audio', source=source)

    def append_video(self,
                     video_uri: str,
                     source: Optional[str] = None):
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
                f"Invalid video URI: '{video_uri}' (must have a valid video file extension)")

        # Append the video block:
        self.append_uri(video_uri, 'video', source=source)

    def append_document(self,
                        document_uri: str,
                        source: Optional[str] = None):
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
        if not any(
                document_uri.endswith(ext) for ext in document_file_extensions):
            # If it is a remote URL, then it is ok to not see a correct extension!
            if document_uri.startswith('http'):
                pass
            else:
                raise ValueError(
                    f"Invalid document URI: '{document_uri}' (must have a valid document file extension)")

        # Append the document block:
        self.append_uri(document_uri, 'document', source=source)

    def append_table(self,
                     table: Union[str, 'ndarray', 'DataFrame'],
                     source: Optional[str] = None):
        """
        Append a table to the message.

        Parameters
        ----------
        table : Union[str, ndarray, DataFrame]
            The table to append. Can be a URI of a local or remote file, a pandas DataFrame, or a numpy array.
        source : Optional[str]
            The source of the table (e.g., file path or url).
        """

        # By default the source is None:
        source = None

        # If table is an URI of tabel file (csv, tsv, xls, xlsx, etc) download file and load it as a pandas DataFrame:
        if isinstance(table, str):
            # Check that it is a valid table URI:
            if not table.startswith('http') and not table.startswith('file'):
                raise ValueError(
                    f"Invalid table URI: '{table}' (must start with 'http' or 'file')")

            # Check that the file extension is valid:
            if not any(table.endswith(ext) for ext in table_file_extensions):
                raise ValueError(
                    f"Invalid table URI: '{table}' (must have a valid table file extension)")

            # source is:
            source = table

            # Download the table file to a local temp file using download_table_to_temp_file:
            table = download_table_to_temp_file(table)

            # Detect character encoding:
            import chardet
            with open(table, "rb") as f:
                result = chardet.detect(f.read())

            # Download the table file taking care of the character encoding and errors
            from pandas import read_csv
            table = read_csv(table, encoding=result['encoding'])

        # Check that it is a numpy array or a pandas DataFrame:
        from numpy import ndarray
        from pandas import DataFrame
        if not (isinstance(table, ndarray) or isinstance(table, DataFrame)):
            raise ValueError(
                f"Table must be a numpy array or a pandas DataFrame, not {type(table)}")

        # Append the table block:
        self.append_block(
            MessageBlock(block_type='table', content=table, source=source))

    def append_folder(self,
                      folder: str,
                      depth: Optional[int] = None,
                      all_archive_files: bool = False):
        """
        Append a folder to the message.

        Parameters
        ----------
        folder : str
            The folder to append.
        depth : Optional[int]
            The depth to traverse the folder.
            If None then there is no depth limits (default: None).
        all_archive_files: bool
            Whether to include all files in archives,
            disregarding the depth of files in the archives (default: False).
        """

        self.append_text("\n##### Contents of folder: " + folder + "\n")

        # recursively traverse folder and enumerate all files up to specified depth:
        for root, dirs, files in os.walk(folder):

            # append files to the message:
            for file in files:

                # file path within folder:
                file_path = os.path.join(root, file)

                # Make a local file URi from the file path:
                file_uri = 'file://' + file_path

                if any(file.endswith(ext) for ext in image_file_extensions):
                    self.append_text("\n###### Image file: " + file + "\n")
                    self.append_image(file_uri, source=file_path)
                elif any(file.endswith(ext) for ext in audio_file_extensions):
                    self.append_text("\n###### Audio file: " + file + "\n")
                    self.append_audio(file_uri, source=file_path)
                elif any(file.endswith(ext) for ext in video_file_extensions):
                    self.append_text("\n###### Video file: " + file + "\n")
                    self.append_video(file_uri, source=file_path)
                elif any(
                        file.endswith(ext) for ext in document_file_extensions):
                    self.append_text("\n###### Document file: " + file + "\n")
                    self.append_document(file_uri, source=file_path)
                elif any(file.endswith(ext) for ext in archive_file_extensions):
                    depth = None if all_archive_files or not depth else depth - 1
                    self.append_archive(file_uri, depth)

            # append subfolders to the message:
            if not depth or depth >= 1:
                for dir in dirs:
                    self.append_folder(dir, depth - 1)

    def append_archive(self,
                       archive: str,
                       depth: int = 1):
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
        self.append_text(
            f"\n##### Contents of archive: {archive} decompressed into folder: " + temp_folder + "\n")

        # append the extracted folder to the message:
        self.append_folder(temp_folder, depth)

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
        if 0 <= index < len(self.blocks):
            return self.blocks[index]
        return None

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
            message_string += str(block) + '\n'
        return message_string

    def lower(self) -> str:
        """
        Return the message as a string in lowercase.
        Returns
        -------
        str
            The message as a string in lowercase.
        """
        return str(self).lower()

    def __repr__(self) -> str:
        """
        Return the message as a string.
        Returns
        -------
        str
            The message as a string.
        """
        return str(self)
