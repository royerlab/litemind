from typing import Optional

from litemind.media.media_uri import MediaURI
from litemind.media.types.media_text import Text
from litemind.utils.file_types.file_types import is_text_file, probe


class File(MediaURI):
    """Media for generic files not covered by other media types.

    Acts as a catch-all for file types that do not have a dedicated media
    class (e.g. binaries, archives). Provides conversion to a descriptive
    Text media including file metadata and content previews.
    """

    def __init__(self, uri: str, extension: Optional[str] = None, **kwargs):
        """Create a new file media.

        Parameters
        ----------
        uri : str
            The file URI or local file path.
        extension : str, optional
            File extension override without the leading dot (e.g. ``"exe"``).
        **kwargs
            Additional keyword arguments forwarded to ``MediaURI``.
        """

        super().__init__(uri=uri, extension=extension, **kwargs)

    def load_from_uri(self):
        """Load the raw file content from the URI.

        Populates ``self.data`` with the file content as bytes.

        Returns
        -------
        bytes
            The raw file content.
        """
        local_path = self.to_local_file_path()
        with open(local_path, "rb") as f:
            self.data = f.read()

        return self.data

    def to_markdown_text_media(self, hex_dump_length=128) -> Text:
        """Create an LLM-friendly text description of the file.

        Includes file metadata (size, type, MIME type, last modified date)
        and either the full text content (for text files) or hex dumps of
        the first and last bytes (for binary files).

        Parameters
        ----------
        hex_dump_length : int, optional
            Number of bytes to include in the hex dump from the beginning
            and end of binary files. Default is 128.

        Returns
        -------
        Text
            A Text media containing the file description.
        """
        import binascii
        import os

        # Return a local file path:
        local_file_path = self.to_local_file_path()

        # Get file info:
        file_info = probe(local_file_path)

        # Get file size and other attributes
        try:
            file_size = os.path.getsize(local_file_path)
            abs_path = os.path.abspath(local_file_path)
            last_modified = os.path.getmtime(local_file_path)
            from datetime import datetime

            last_modified_str = datetime.fromtimestamp(last_modified).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Collect file metadata
            description = [
                f"Description of file: '{self.get_filename()}':",
                f"- Absolute Path: {abs_path}",
                f"- File Size: {file_size} bytes ({file_size / 1024:.2f} KB)",
                f"- Last Modified: {last_modified_str}",
                f"- File Type: {file_info.get('file_type', 'Unknown')}",
                f"- MIME Type: {file_info.get('mime_type', 'Unknown')}",
            ]

            # Add any additional info from file_info that might be useful
            for key, value in file_info.items():
                if key not in ["file_type", "mime_type"] and value is not None:
                    description.append(f"- {key}: {value}")

            # if the file is empty, return early
            if file_size == 0:
                description.append("\nFile is empty.")
                return Text(text="\n".join(description))

            # if the file is a text file, read its content and add it to the description:
            elif is_text_file(local_file_path):
                with open(local_file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    description.append("\nFile content:")
                    description.append(file_content)
                    return Text(text="\n".join(description))

            # If the file is not a text file, we will read a hex dump of the file content
            else:

                # Read beginning and end of file
                with open(local_file_path, "rb") as f:
                    # If file is small, just read the whole thing
                    if file_size <= hex_dump_length * 2:
                        file_bytes = f.read()
                        file_hex = binascii.hexlify(file_bytes).decode("ascii")

                        description.append(
                            f"\nEntire file content ({file_size} bytes):"
                        )
                        for i in range(0, len(file_hex), 32):
                            hex_line = " ".join(
                                file_hex[i : i + 32][j : j + 2]
                                for j in range(0, min(32, len(file_hex[i : i + 32])), 2)
                            )
                            description.append(hex_line)
                    else:
                        # Read first bytes
                        start_bytes = f.read(hex_dump_length)

                        # Go to end and read last bytes
                        f.seek(max(0, file_size - hex_dump_length))
                        end_bytes = f.read(hex_dump_length)

                        # Convert to hex representation with byte grouping
                        start_hex = binascii.hexlify(start_bytes).decode("ascii")
                        end_hex = binascii.hexlify(end_bytes).decode("ascii")

                        # Format hex dump in readable format (grouped by bytes)
                        description.append(f"\nFirst {hex_dump_length} bytes (hex):")
                        for i in range(0, len(start_hex), 32):
                            hex_line = " ".join(
                                start_hex[i : i + 32][j : j + 2]
                                for j in range(
                                    0, min(32, len(start_hex[i : i + 32])), 2
                                )
                            )
                            description.append(hex_line)

                        description.append(f"\nLast {hex_dump_length} bytes (hex):")
                        for i in range(0, len(end_hex), 32):
                            hex_line = " ".join(
                                end_hex[i : i + 32][j : j + 2]
                                for j in range(0, min(32, len(end_hex[i : i + 32])), 2)
                            )
                            description.append(hex_line)

        except Exception as e:
            description = [f"Error analyzing file: {str(e)}"]

        # Create a Text media with the description
        text_content = "\n".join(description)
        return Text(text=text_content)
