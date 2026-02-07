"""
Repository export command.

Provides the ``export`` CLI subcommand which concatenates all matching source
files in a repository into a single text file suitable for ingestion by an LLM.
"""

import os
from typing import Optional

from arbol import aprint, asection

from litemind.agent.messages.message import Message
from litemind.tools.commands.utils import default_folder_scanning_parameters


def export_repo(
    folder_path: str,
    allowed_extensions: Optional[list[str]] = None,
    excluded_files: Optional[list[str]] = None,
    output_file: Optional[str] = None,
) -> str:
    """
    Export an entire repository as a single concatenated text file.

    Walks the directory tree rooted at *folder_path*, selects files whose
    extensions match *allowed_extensions*, and writes their contents into
    *output_file*. The output file itself is automatically excluded from
    the export.

    Parameters
    ----------
    folder_path : str
        Root directory of the repository to export.
    allowed_extensions : Optional[list[str]]
        File extensions to include. If ``None``, sensible defaults are
        applied (see ``default_folder_scanning_parameters``).
    excluded_files : Optional[list[str]]
        File or directory names to skip. If ``None``, sensible defaults
        are applied.
    output_file : Optional[str]
        Destination path for the exported text. Defaults to
        ``"exported.txt"`` when called from the CLI.

    Returns
    -------
    str
        The concatenated repository contents that were written to
        *output_file*.
    """
    # Apply default output file if not specified:
    if output_file is None:
        output_file = "exported.txt"

    with asection(
        f"Exporting entire repository in {folder_path} to single file: {output_file}"
    ):
        # Get output_file's base name:
        output_file_name = os.path.basename(output_file)

        # Get the default folder scanning parameters:
        allowed_extensions, excluded_files = default_folder_scanning_parameters(
            allowed_extensions, excluded_files
        )

        # Add output_file_name to excluded_files:
        excluded_files.append(output_file_name)

        # Print the parameters:
        aprint(f"Excluded files: {', '.join(excluded_files)}")
        aprint(f"Allowed extensions: {', '.join(allowed_extensions)}")

        # Create a message with the prompt:
        message = Message(role="user")

        # Append the prompt and the folder contents to the message:
        message.append_folder(
            folder_path,
            allowed_extensions=allowed_extensions,
            excluded_files=excluded_files,
        )

        # Convert the message to a string
        message_str = str(message)

        # Save string to file:
        with open(output_file, "w") as file:
            file.write(message_str)

        # Return the whole folder contents as a string:
        return message_str
