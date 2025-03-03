from typing import Optional

from arbol import asection

from litemind.agent.messages.message import Message
from litemind.apis.base_api import BaseApi
from litemind.tools.commands.utils import default_folder_scanning_parameters


def export_repo(
    folder_path: str,
    allowed_extensions: Optional[list[str]] = None,
    excluded_files: Optional[list[str]] = None,
    output_file: Optional[str] = None,
) -> str:
    """
    Exports the whole repository as a single file.

    Parameters
    ----------
    folder_path: str
        The path to the folder containing the Python repository.
    allowed_extensions: Optional[list[str]]
        The list of allowed extensions for files to include in the README.
    excluded_files: Optional[list[str]]
        The list of files to exclude from the README.
    output_file: Optional[str]
        The path to the file to save the entire repository to.

    Returns
    -------
    str
        The whole repo as a string

    """
    with asection(f"Exporting entire repository in {folder_path} to single file"):

        allowed_extensions, excluded_files = default_folder_scanning_parameters(
            allowed_extensions, excluded_files
        )

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
