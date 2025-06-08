import os
import tempfile
from pathlib import Path

import pytest

from litemind.agent.messages.message import Message


@pytest.fixture
def test_folder():
    """Create a temporary folder structure for testing."""
    with tempfile.TemporaryDirectory(delete=False) as tmp_dir:
        # Create a folder structure for testing
        # Main folder
        folder = Path(tmp_dir) / "test_folder"
        folder.mkdir()

        # Create a text file
        (folder / "test.txt").write_text("This is a test file")

        # Create an empty file
        (folder / "empty.txt").touch()

        # Create a Python file
        (folder / "script.py").write_text("print('Hello, World!')")

        # Create a hidden file
        (folder / ".hidden").write_text("Hidden content")

        # Create a subfolder
        subfolder = folder / "subfolder"
        subfolder.mkdir()
        (subfolder / "subfile.txt").write_text("This is a file in a subfolder")

        # Create a hidden subfolder
        hidden_subfolder = folder / ".hidden_folder"
        hidden_subfolder.mkdir()
        (hidden_subfolder / "hidden_file.txt").write_text("This is in a hidden folder")

        yield str(folder)  # Return the path as a string


def test_append_folder_basic(test_folder):
    """Test basic functionality of append_folder method."""
    message = Message(role="user")
    message.append_folder(test_folder)

    # Check that the message contains blocks
    assert len(message.blocks) > 0

    # Check that the directory structure was added
    assert "Directory structure:" in str(message)

    # Check that file content was added
    assert "test.txt" in str(message)
    assert "/subfolder/subfile.txt" in str(message)


def test_append_folder_depth_limit(test_folder):
    """Test folder traversal with depth limit."""
    message = Message(role="user")
    # Set depth to 0 to only include the top level
    message.append_folder(test_folder, depth=1)

    # Should include top level files
    assert "test.txt" in str(message)

    # Should not include files from subfolder
    assert "subfile.txt" not in str(message)


def test_append_folder_allowed_extensions(test_folder):
    """Test folder traversal with allowed extensions filter."""
    message = Message(role="user")
    message.append_folder(test_folder, allowed_extensions=[".py"])

    # Should include Python files
    assert "script.py" in str(message)

    # Should not include text files
    assert "test.txt" not in str(message)
    assert "subfile.txt" not in str(message)


def test_append_folder_excluded_files(test_folder):
    """Test folder traversal with excluded files."""
    message = Message(role="user")
    message.append_folder(test_folder, excluded_files=["test.txt"])

    # Should not include excluded files
    assert "test.txt" not in str(message)

    # Should include other files
    assert "script.py" in str(message)


def test_append_folder_include_hidden_files(test_folder):
    """Test folder traversal with hidden files included."""
    # Test with hidden files excluded (default)
    message_without_hidden = Message(role="user")
    message_without_hidden.append_folder(test_folder)
    assert ".hidden" not in str(message_without_hidden)

    # Test with hidden files included
    message_with_hidden = Message(role="user")
    message_with_hidden.append_folder(test_folder, include_hidden_files=True)
    assert ".hidden" in str(message_with_hidden)


def test_append_folder_empty_file(test_folder):
    """Test handling of empty files."""
    message = Message(role="user")
    message.append_folder(test_folder)

    # Should mention that the file is empty
    assert "empty.txt" in str(message)
    assert "Empty" in str(message)


def test_append_folder_nonexistent_folder():
    """Test handling of non-existent folders."""
    message = Message(role="user")
    non_existent_path = "/path/that/does/not/exist"
    message.append_folder(non_existent_path)

    # Should include a message about the non-existent folder
    assert f"Folder '{non_existent_path}' does not exist" in str(message)


def test_append_folder_empty_folder():
    """Test handling of empty folders."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create an empty folder
        empty_folder = Path(tmp_dir) / "empty_folder"
        empty_folder.mkdir()

        message = Message(role="user")
        message.append_folder(str(empty_folder))

        # Should include a message about the empty folder
        assert f"Folder '{empty_folder}' is empty" in str(message)


def test_append_folder_file_path():
    """Test handling when a file path is provided instead of a folder."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(b"Test content")
        tmp_file_path = tmp_file.name
        tmp_file.close()

        try:
            message = Message(role="user")
            message.append_folder(tmp_file_path)

            # Should include a message that the path is not a folder
            assert f"File '{tmp_file_path}' is not a folder" in str(message)
        finally:
            os.unlink(tmp_file_path)


def test_append_folder_with_code_files(test_folder):
    """Test handling of code files."""
    message = Message(role="user")
    message.append_folder(test_folder)

    # Should identify Python files as code
    assert "│   └── subfile.txt (29 bytes)" in str(message)
    assert "Empty File: empty.txt" in str(message)
    assert "Prog. Lang. Code File: script.py" in str(message)
