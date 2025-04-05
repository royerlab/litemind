from litemind.utils.folder_description import (
    file_info_header,
    generate_tree_structure,
    human_readable_size,
    read_binary_file_info,
    read_file_content,
)


def test_human_readable_size():
    assert human_readable_size(500) == "500 bytes"
    assert human_readable_size(2048) == "2.00 kilobytes"
    assert human_readable_size(1048576) == "1.00 megabytes"
    assert human_readable_size(1073741824) == "1.00 gigabytes"
    assert human_readable_size(1099511627776) == "1.00 terabytes"
    assert human_readable_size(1125899906842624) == "1.00 petabytes"


def test_generate_tree_structure(tmp_path):
    # Create a temporary directory structure
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "file.txt").write_text("content")
    (tmp_path / "folder" / "subfolder").mkdir()
    (tmp_path / "folder" / "subfolder" / "file2.txt").write_text("content")

    expected_tree = (
        "├── file.txt (7 bytes)\n" "└── subfolder/\n" "    └── file2.txt (7 bytes)\n"
    )

    assert generate_tree_structure(str(tmp_path / "folder")) == expected_tree


def test_read_file_content(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("This is a test file.")
    assert read_file_content(file_path) == "This is a test file."


def test_read_binary_file_info(tmp_path):
    file_path = tmp_path / "file.bin"
    file_path.write_bytes(b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09")
    size, hex_content = read_binary_file_info(file_path)
    assert size == 10
    assert hex_content == "00010203040506070809"


def test_file_info_header(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("This is a test file.")
    header = file_info_header(file_path, "Text")
    assert "Text File: file.txt" in header
    assert "Size: 20 bytes" in header
    assert "Last Modified:" in header
    assert "Created:" in header
