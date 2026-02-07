import os
import tempfile

from litemind.tools.commands.export_repo import export_repo


def test_export_repo_default_output_file():
    """Test that export_repo uses 'exported.txt' when output_file is None.

    Regression test: passing output_file=None used to crash with TypeError
    on os.path.basename(None).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple file to export
        test_file = os.path.join(temp_dir, "hello.py")
        with open(test_file, "w") as f:
            f.write("print('hello')\n")

        # Call with output_file=None (the default)
        result = export_repo(folder_path=temp_dir, output_file=None)

        # Should succeed and return non-empty content
        assert isinstance(result, str)
        assert "hello" in result
