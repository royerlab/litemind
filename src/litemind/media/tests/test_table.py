import os
import tempfile

import pandas as pd
import pytest

from litemind.media.types.media_table import Table
from litemind.media.types.media_text import Text


def test_table_initialization_with_uri():
    """Test that the Table is correctly initialized with a URI."""
    table = Table(uri="file://test.csv")
    assert table.uri == "file://test.csv"
    assert table.dataframe is None


def test_table_initialization_with_dataframe():
    """Test that the Table is correctly initialized with a DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table = Table.from_dataframe(df)
    table.load_from_uri()
    assert table.dataframe.equals(df)
    assert table.uri is not None


def test_table_initialization_no_uri_or_dataframe():
    """Test that the Table raises ValueError when neither URI nor DataFrame is provided."""
    with pytest.raises(TypeError):
        Table()


def test_from_dataframe():
    """Test that the Table is correctly created from a DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table = Table.from_dataframe(df)
    assert table.uri is not None
    assert table.dataframe is None


def test_from_dataframe_with_filepath():
    """Test that the Table is correctly created from a DataFrame with a specified filepath."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
        filepath = tmp_file.name
        table = Table.from_dataframe(df, filepath=filepath)
        assert table.uri == f"file://{filepath}"
        assert table.dataframe is None
        os.remove(filepath)  # Clean up the temporary file


def test_load_from_uri(tmp_path):
    """Test that the table data is correctly loaded from the URI."""
    # Create a temporary CSV file for testing
    csv_data = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_data)
    uri = f"file://{file_path}"

    table = Table(uri=uri)
    table.load_from_uri()

    # Assert that the dataframe is loaded correctly
    assert table.dataframe is not None
    expected_df = pd.read_csv(file_path)
    pd.testing.assert_frame_equal(table.dataframe, expected_df)


def test_load_from_uri_no_uri():
    """Test that ValueError is raised when no URI is provided."""
    table = Table.from_dataframe(pd.DataFrame({"col1": [1, 2], "col2": [3, 4]}))
    table.uri = None
    with pytest.raises(ValueError, match="No URI provided to load the table data"):
        table.load_from_uri()


def test_to_markdown_with_uri(tmp_path):
    """Test that the table is correctly converted to a markdown string when initialized with a URI."""
    # Create a temporary CSV file for testing
    csv_data = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_data)
    uri = f"file://{file_path}"

    table = Table(uri=uri)
    markdown_string = table.to_markdown()

    assert isinstance(markdown_string, str)
    assert "```dataframe" in markdown_string
    assert "|   col1 |   col2 |" in markdown_string
    assert "|-------:|-------:|" in markdown_string
    assert "|      1 |      2 |" in markdown_string
    assert "|      3 |      4 |" in markdown_string


def test_to_markdown_with_dataframe():
    """Test that the table is correctly converted to a markdown string when initialized with a DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table = Table.from_dataframe(df)
    markdown_string = table.to_markdown()

    assert isinstance(markdown_string, str)
    assert "```dataframe" in markdown_string
    assert "|   col1 |   col2 |" in markdown_string
    assert "|-------:|-------:|" in markdown_string
    assert " 1 " in markdown_string
    assert " 4 " in markdown_string


def test_to_markdown_text_media_with_dataframe():
    """Test that the table is correctly converted to a markdown Text media when initialized with a DataFrame."""
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    table = Table.from_dataframe(df)
    text_media = table.to_markdown_text_media()

    assert isinstance(text_media, Text)
    assert "```dataframe" in text_media.text
    assert "|   col1 |   col2 |" in text_media.text
    assert "|-------:|-------:|" in text_media.text
    assert " 1 " in text_media.text
    assert " 4 " in text_media.text


def test_to_markdown_text_media_with_uri(tmp_path):
    """Test that the table is correctly converted to a markdown Text media when initialized with a URI."""
    # Create a temporary CSV file for testing
    csv_data = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_data)
    uri = f"file://{file_path}"

    table = Table(uri=uri)
    text_media = table.to_markdown_text_media()

    assert isinstance(text_media, Text)
    assert "```dataframe" in text_media.text
    assert "|   col1 |   col2 |" in text_media.text
    assert "|-------:|-------:|" in text_media.text
    assert " 1 " in text_media.text
    assert " 4 " in text_media.text
